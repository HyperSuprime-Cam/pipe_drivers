import numpy

import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.meas.algorithms as measAlg

from lsst.afw.cameraGeom.utils import makeImageFromCamera
from lsst.pipe.base import ArgumentParser, Struct
from lsst.pex.config import Config, Field, ConfigurableField, ConfigField
from lsst.ctrl.pool.pool import Pool
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.pipe.drivers.background import SkyMeasurementTask, FocalPlaneBackground, FocalPlaneBackgroundConfig


DEBUG = False  # Debugging outputs?
BINNING = 8  # Binning factor for debugging outputs


def makeCameraImage(camera, exposures, filename, binning=8):
    """Make and write an image of an entire focal plane

    Parameters
    ----------
    camera : `lsst.afw.cameraGeom.Camera`
        Camera description.
    exposures : `dict` mapping detector ID to `lsst.afw.image.Exposure`
        CCD exposures, binned by `binning`.
    filename : `str`
        Output filename.
    binning : `int`
        Binning size that has been applied to images.
    """
    class ImageSource(object):
        """Source of images for makeImageFromCamera"""
        def __init__(self, exposures):
            """Constructor

            Parameters
            ----------
            exposures : `dict` mapping detector ID to `lsst.afw.image.Exposure`
                CCD exposures, already binned.
            """
            self.isTrimmed = True
            self.exposures = exposures
            self.background = numpy.nan

        def getCcdImage(self, detector, imageFactory, binSize):
            """Provide image of CCD to makeImageFromCamera"""
            if detector.getId() not in self.exposures:
                return imageFactory(1, 1)
            image = self.exposures[detector.getId()]
            if hasattr(image, "getMaskedImage"):
                image = image.getMaskedImage()
            if hasattr(image, "getMask"):
                mask = image.getMask()
                isBad = mask.getArray() & mask.getPlaneBitMask("NO_DATA") > 0
                image = image.clone()
                image.getImage().getArray()[isBad] = numpy.nan
            if hasattr(image, "getImage"):
                image = image.getImage()
            return image

    image = makeImageFromCamera(
        camera,
        imageSource=ImageSource(dict(exp for exp in exposures if exp is not None)),
        imageFactory=afwImage.ImageF,
        binSize=binning
    )
    image.writeFits(filename)


class SkyCorrectionConfig(Config):
    """Configuration for SkyCorrectionTask"""
    bgModel1 = ConfigField(dtype=FocalPlaneBackgroundConfig, doc="First background model")
    bgModel2 = ConfigField(dtype=FocalPlaneBackgroundConfig, doc="Second background model")
    sky = ConfigurableField(target=SkyMeasurementTask, doc="Sky measurement")
    detection = ConfigurableField(target=measAlg.SourceDetectionTask, doc="Detection configuration")
    detectSigma = Field(dtype=float, default=2.0, doc="Detection PSF gaussian sigma")
    subtractBackground = ConfigurableField(target=measAlg.SubtractBackgroundTask,
                                           doc="Background configuration")

    doBgModel1 = Field(dtype=bool, default=True, doc="Do first background model subtraction?")
    doSky = Field(dtype=bool, default=True, doc="Do sky frame subtraction?")
    doBgModel2 = Field(dtype=bool, default=True, doc="Do second background model subtraction?")

    def setDefaults(self):
        Config.setDefaults(self)
        self.bgModel2.xSize = 512
        self.bgModel2.ySize = 512


class SkyCorrectionTask(BatchPoolTask):
    """Correct sky over entire focal plane"""
    ConfigClass = SkyCorrectionConfig
    _DefaultName = "skyCorr"

    def __init__(self, *args, **kwargs):
        BatchPoolTask.__init__(self, *args, **kwargs)
        self.makeSubtask("sky")
        self.makeSubtask("detection")
        self.makeSubtask("subtractBackground")

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = ArgumentParser(name="skyCorr", *args, **kwargs)
        parser.add_id_argument("--id", datasetType="calexp", level="visit",
                               help="data ID, e.g. --id visit=12345")
        return parser

    def run(self, expRef):
        """Perform sky correction on an exposure

        We restore the original sky, and remove it again using multiple
        algorithms. We optionally apply:

        1. A large-scale background model.
        2. A sky frame.
        3. A small-scale background model.

        Only the master node executes this method. The data is held on
        the slave nodes, which do all the hard work.

        Parameters
        ----------
        expRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.
        """
        with self.logOperation("processing %s" % (expRef.dataId,)):
            pool = Pool()
            pool.cacheClear()
            pool.storeSet(butler=expRef.getButler())
            camera = expRef.get("camera")

            dataIdList = [ccdRef.dataId for ccdRef in expRef.subItems("ccd") if
                          ccdRef.datasetExists("calexp")]

            exposures = pool.map(self.loadImage, dataIdList)
            if DEBUG:
                makeCameraImage(camera, exposures, "restored.fits")
                exposures = pool.mapToPrevious(self.collectOriginal, dataIdList)
                makeCameraImage(camera, exposures, "original.fits")

            if self.config.doBgModel1:
                bgModel = FocalPlaneBackground.fromCamera(self.config.bgModel1, camera)
                data = [Struct(dataId=dataId, bgModel=bgModel.clone()) for dataId in dataIdList]
                bgModelList = pool.mapToPrevious(self.accumulateModel, data)
                for bg in bgModelList:
                    bgModel.merge(bg)

                if DEBUG:
                    bgModel.getStatsImage().writeFits("bgModel.fits")
                    self.log.info("Background model 1: %s" % (bgModel.getStatsImage().getArray(),))
                exposures = pool.mapToPrevious(self.subtractModel, dataIdList, bgModel)
                if DEBUG:
                    makeCameraImage(camera, exposures, "modelsub.fits")

            if self.config.doSky:
                measScales = pool.mapToPrevious(self.measureSkyFrame, dataIdList)
                scale = self.sky.solveScales(measScales)
                self.log.info("Sky frame scale: %s" % (scale,))
                exposures = pool.mapToPrevious(self.subtractSkyFrame, dataIdList, scale)
                if DEBUG:
                    makeCameraImage(camera, exposures, "skysub.fits")

            if self.config.doBgModel2:
                bgModel = FocalPlaneBackground.fromCamera(self.config.bgModel2, camera)
                data = [Struct(dataId=dataId, bgModel=bgModel.clone()) for dataId in dataIdList]
                bgModelList = pool.mapToPrevious(self.accumulateModel, data)
                for bg in bgModelList:
                    bgModel.merge(bg)
                if DEBUG:
                    self.log.info("Background model 2: %s" % (bgModel.getStatsImage().getArray(),))
                    bgModel.getStatsImage().writeFits("bgModel.fits")
                exposures = pool.mapToPrevious(self.subtractModel, dataIdList, bgModel)

            if DEBUG:
                makeCameraImage(camera, exposures, "final.fits")

            pool.mapToPrevious(self.write, dataIdList)

    def loadImage(self, cache, dataId):
        """Load original image and restore the sky

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Resultant exposure.
        """
        cache.dataId = dataId
        cache.exposure = cache.butler.get("calexp", dataId, immediate=True).clone()
        cache.bgList = afwMath.BackgroundList()  # Empty because we're restoring the original background
        bgOld = cache.butler.get("calexpBackground", dataId, immediate=True)
        image = cache.exposure.getMaskedImage()
        image += bgOld.getImage()
        return self.collect(cache)

    def measureSkyFrame(self, cache, dataId):
        """Measure scale for sky frame

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.

        Returns
        -------
        scale : `float`
            Scale for sky frame.
        """
        assert cache.dataId == dataId
        cache.sky = self.sky.getSkyData(cache.butler, dataId)
        scale = self.sky.measureScale(cache.exposure.getMaskedImage(), cache.sky)
        return scale

    def subtractSkyFrame(self, cache, dataId, scale):
        """Subtract sky frame

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.
        scale : `float`
            Scale for sky frame.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Resultant exposure.
        """
        assert cache.dataId == dataId
        self.sky.subtractSkyFrame(cache.exposure.getMaskedImage(), cache.sky, scale, cache.bgList)
        return self.collect(cache)

    def accumulateModel(self, cache, data):
        """Fit background model for CCD

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        data : `lsst.pipe.base.Struct`
            Data identifier, with `dataId` (data identifier) and `bgModel`
            (background model) elements.

        Returns
        -------
        bgModel : `lsst.pipe.drivers.background.FocalPlaneBackground`
            Background model.
        """
        assert cache.dataId == data.dataId
        data.bgModel.addCcd(cache.exposure)
        return data.bgModel

    def subtractModel(self, cache, dataId, bgModel):
        """Subtract background model

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.
        bgModel : `lsst.pipe.drivers.background.FocalPlaneBackround`
            Background model.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Resultant exposure.
        """
        assert cache.dataId == dataId
        exposure = cache.exposure
        image = exposure.getMaskedImage()
        detector = exposure.getDetector()
        bbox = image.getBBox()
        cache.bgModel = bgModel.toCcdBackground(detector, bbox)
        image -= cache.bgModel.getImage()
        cache.bgList.append(cache.bgModel[0])
        return self.collect(cache)

    def collect(self, cache):
        """Collect exposure for potential visualisation

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.

        Returns
        -------
        detId : `int`
            Detector identifier.
        image : `lsst.afw.image.MaskedImage`
            Binned image.
        """
        return (cache.exposure.getDetector().getId(),
                afwMath.binImage(cache.exposure.getMaskedImage(), BINNING))

    def collectOriginal(self, cache, dataId):
        """Collect original image for visualisation

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.

        Returns
        -------
        detId : `int`
            Detector identifier.
        image : `lsst.afw.image.MaskedImage`
            Binned image.
        """
        exposure = cache.butler.get("calexp", dataId, immediate=True)
        return (exposure.getDetector().getId(),
                afwMath.binImage(exposure.getMaskedImage(), BINNING))

    def write(self, cache, dataId):
        """Write resultant exposure

        This method runs on the slave nodes.

        WARNING: We clobber the calexp in the data repository! This may not
        be desirable, but nor do we want to introduce multiple datasets that
        the user has to select down the road.  The user should write to a
        different rerun or output data repository.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.
        """
        cache.butler.put(cache.exposure, "calexp", dataId)
        cache.butler.put(cache.bgList, "calexpBackground", dataId)

    def _getMetadataName(self):
        """There's no metadata to write out"""
        return None
