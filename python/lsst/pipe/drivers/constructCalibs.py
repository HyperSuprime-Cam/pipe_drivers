from __future__ import absolute_import, division, print_function
import sys
import math
import time
import argparse
import traceback

import numpy as np
from builtins import zip
from builtins import range

from lsst.pex.config import Config, ConfigurableField, Field, ListField
from lsst.pipe.base import Task, Struct, TaskRunner, ArgumentParser
import lsst.daf.base as dafBase
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDet
import lsst.afw.image as afwImage
import lsst.meas.algorithms as measAlg
from lsst.pipe.tasks.repair import RepairTask
from lsst.ip.isr import IsrTask
from lsst.afw.cameraGeom.utils import makeImageFromCamera

from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, NODE
from lsst.pipe.drivers.background import SkyMeasurementTask, FocalPlaneBackground, FocalPlaneBackgroundConfig

from .checksum import checksum
from .utils import getDataRef


class CalibStatsConfig(Config):
    """Parameters controlling the measurement of background statistics"""
    stat = Field(doc="Statistic to use to estimate background (from lsst.afw.math)", dtype=int,
                 default=afwMath.MEANCLIP)
    clip = Field(doc="Clipping threshold for background",
                 dtype=float, default=3.0)
    nIter = Field(doc="Clipping iterations for background",
                  dtype=int, default=3)
    mask = ListField(doc="Mask planes to reject",
                     dtype=str, default=["DETECTED", "BAD", "NO_DATA",])


class CalibStatsTask(Task):
    """Measure statistics on the background

    This can be useful for scaling the background, e.g., for flats and fringe frames.
    """
    ConfigClass = CalibStatsConfig

    def run(self, exposureOrImage):
        """!Measure a particular statistic on an image (of some sort).

        @param exposureOrImage    Exposure, MaskedImage or Image.
        @return Value of desired statistic
        """
        stats = afwMath.StatisticsControl(self.config.clip, self.config.nIter,
                                          afwImage.MaskU.getPlaneBitMask(self.config.mask))
        try:
            image = exposureOrImage.getMaskedImage()
        except:
            try:
                image = exposureOrImage.getImage()
            except:
                image = exposureOrImage

        return afwMath.makeStatistics(image, self.config.stat, stats).getValue()


class CalibCombineConfig(Config):
    """Configuration for combining calib images"""
    rows = Field(doc="Number of rows to read at a time",
                 dtype=int, default=512)
    mask = ListField(doc="Mask planes to respect", dtype=str,
                     default=["SAT", "DETECTED", "INTRP"])
    combine = Field(doc="Statistic to use for combination (from lsst.afw.math)", dtype=int,
                    default=afwMath.MEANCLIP)
    clip = Field(doc="Clipping threshold for combination",
                 dtype=float, default=3.0)
    nIter = Field(doc="Clipping iterations for combination",
                  dtype=int, default=3)
    stats = ConfigurableField(target=CalibStatsTask,
                              doc="Background statistics configuration")


class CalibCombineTask(Task):
    """Task to combine calib images"""
    ConfigClass = CalibCombineConfig

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.makeSubtask("stats")

    def run(self, sensorRefList, expScales=None, finalScale=None, inputName="postISRCCD"):
        """!Combine calib images for a single sensor

        @param sensorRefList   List of data references to combine (for a single sensor)
        @param expScales       List of scales to apply for each exposure
        @param finalScale      Desired scale for final combined image
        @param inputName       Data set name for inputs
        @return combined image
        """
        width, height = self.getDimensions(sensorRefList)
        maskVal = 0
        for mask in self.config.mask:
            maskVal |= afwImage.MaskU.getPlaneBitMask(mask)
        stats = afwMath.StatisticsControl(
            self.config.clip, self.config.nIter, maskVal)

        # Combine images
        combined = afwImage.MaskedImageF(width, height)
        numImages = len(sensorRefList)
        imageList = [None]*numImages
        for start in range(0, height, self.config.rows):
            rows = min(self.config.rows, height - start)
            box = afwGeom.Box2I(afwGeom.Point2I(0, start),
                                afwGeom.Extent2I(width, rows))
            subCombined = combined.Factory(combined, box)

            for i, sensorRef in enumerate(sensorRefList):
                if sensorRef is None:
                    imageList[i] = None
                    continue
                exposure = sensorRef.get(inputName + "_sub", bbox=box)
                if expScales is not None:
                    self.applyScale(exposure, expScales[i])
                imageList[i] = exposure.getMaskedImage()

            self.combine(subCombined, imageList, stats)

        if finalScale is not None:
            background = self.stats.run(combined)
            self.log.info("%s: Measured background of stack is %f; adjusting to %f" %
                          (NODE, background, finalScale))
            combined *= finalScale / background

        return afwImage.DecoratedImageF(combined.getImage())

    def getDimensions(self, sensorRefList, inputName="postISRCCD"):
        """Get dimensions of the inputs"""
        dimList = []
        for sensorRef in sensorRefList:
            if sensorRef is None:
                continue
            md = sensorRef.get(inputName + "_md")
            dimList.append(afwGeom.Extent2I(
                md.get("NAXIS1"), md.get("NAXIS2")))
        return getSize(dimList)

    def applyScale(self, exposure, scale=None):
        """Apply scale to input exposure

        This implementation applies a flux scaling: the input exposure is
        divided by the provided scale.
        """
        if scale is not None:
            mi = exposure.getMaskedImage()
            mi /= scale

    def combine(self, target, imageList, stats):
        """!Combine multiple images

        @param target      Target image to receive the combined pixels
        @param imageList   List of input images
        @param stats       Statistics control
        """
        images = afwImage.vectorMaskedImageF(
            [img for img in imageList if img is not None])
        afwMath.statisticsStack(target, images, self.config.combine, stats)


def getSize(dimList):
    """Determine a consistent size, given a list of image sizes"""
    dim = set((w, h) for w, h in dimList)
    dim.discard(None)
    if len(dim) != 1:
        raise RuntimeError("Inconsistent dimensions: %s" % dim)
    return dim.pop()


def dictToTuple(dict_, keys):
    """!Return a tuple of specific values from a dict

    This provides a hashable representation of the dict from certain keywords.
    This can be useful for creating e.g., a tuple of the values in the DataId
    that identify the CCD.

    @param dict_  dict to parse
    @param keys  keys to extract (order is important)
    @return tuple of values
    """
    return tuple(dict_[k] for k in keys)


def getCcdIdListFromExposures(expRefList, level="sensor", ccdKeys=["ccd"]):
    """!Determine a list of CCDs from exposure references

    This essentially inverts the exposure-level references (which
    provides a list of CCDs for each exposure), by providing
    a dataId list for each CCD.  Consider an input list of exposures
    [e1, e2, e3], and each exposure has CCDs c1 and c2.  Then this
    function returns:

        {(c1,): [e1c1, e2c1, e3c1], (c2,): [e1c2, e2c2, e3c2]}

    This is a dict whose keys are tuples of the identifying values of a
    CCD (usually just the CCD number) and the values are lists of dataIds
    for that CCD in each exposure.  A missing dataId is given the value
    None.

    @param expRefList   List of data references for exposures
    @param level        Level for the butler to generate CCDs
    @param ccdKeys      DataId keywords that identify a CCD
    @return dict of data identifier lists for each CCD
    """
    expIdList = [[ccdRef.dataId for ccdRef in expRef.subItems(
        level)] for expRef in expRefList]

    # Determine what additional keys make a CCD from an exposure
    ccdKeys = set(ccdKeys)  # Set of keywords in the dataId that identify a CCD
    ccdNames = set()  # Set of tuples which are values for each of the CCDs in an exposure
    for ccdIdList in expIdList:
        for ccdId in ccdIdList:
            name = dictToTuple(ccdId, ccdKeys)
            ccdNames.add(name)

    # Turn the list of CCDs for each exposure into a list of exposures for
    # each CCD
    ccdLists = {}
    for n, ccdIdList in enumerate(expIdList):
        for ccdId in ccdIdList:
            name = dictToTuple(ccdId, ccdKeys)
            if name not in ccdLists:
                ccdLists[name] = []
            ccdLists[name].append(ccdId)

    for ccd in ccdLists:
        ccdLists[ccd] = sorted(ccdLists[ccd], key=lambda dd: dictToTuple(dd, sorted(dd.keys())))

    return ccdLists


def mapToMatrix(pool, func, ccdIdLists, *args, **kwargs):
    """Generate a matrix of results using pool.map

    The function should have the call signature:
        func(cache, dataId, *args, **kwargs)

    We return a dict mapping 'ccd name' to a list of values for
    each exposure.

    @param pool  Process pool
    @param func  Function to call for each dataId
    @param ccdIdLists  Dict of data identifier lists for each CCD name
    @return matrix of results
    """
    dataIdList = sum(ccdIdLists.values(), [])
    resultList = pool.map(func, dataIdList, *args, **kwargs)
    # Piece everything back together
    data = dict((ccdName, [None] * len(expList)) for ccdName, expList in ccdIdLists.items())
    indices = dict(sum([[(tuple(dataId.values()) if dataId is not None else None, (ccdName, expNum))
                         for expNum, dataId in enumerate(expList)]
                        for ccdName, expList in ccdIdLists.items()], []))
    for dataId, result in zip(dataIdList, resultList):
        if dataId is None:
            continue
        ccdName, expNum = indices[tuple(dataId.values())]
        data[ccdName][expNum] = result
    return data


class CalibIdAction(argparse.Action):
    """Split name=value pairs and put the result in a dict"""

    def __call__(self, parser, namespace, values, option_string):
        output = getattr(namespace, self.dest, {})
        for nameValue in values:
            name, sep, valueStr = nameValue.partition("=")
            if not valueStr:
                parser.error("%s value %s must be in form name=value" %
                             (option_string, nameValue))
            output[name] = valueStr
        setattr(namespace, self.dest, output)


class CalibArgumentParser(ArgumentParser):
    """ArgumentParser for calibration construction"""

    def __init__(self, calibName, *args, **kwargs):
        """Add a --calibId argument to the standard pipe_base argument parser"""
        ArgumentParser.__init__(self, *args, **kwargs)
        self.calibName = calibName
        self.add_id_argument("--id", datasetType="raw",
                             help="input identifiers, e.g., --id visit=123 ccd=4")
        self.add_argument("--calibId", nargs="*", action=CalibIdAction, default={},
                          help="identifiers for calib, e.g., --calibId version=1",
                          metavar="KEY=VALUE1[^VALUE2[^VALUE3...]")

    def parse_args(self, *args, **kwargs):
        """Parse arguments

        Checks that the "--calibId" provided works.
        """
        namespace = ArgumentParser.parse_args(self, *args, **kwargs)

        keys = namespace.butler.getKeys(self.calibName)
        parsed = {}
        for name, value in namespace.calibId.items():
            if name not in keys:
                self.error(
                    "%s is not a relevant calib identifier key (%s)" % (name, keys))
            parsed[name] = keys[name](value)
        namespace.calibId = parsed

        return namespace


class CalibConfig(Config):
    """Configuration for constructing calibs"""
    clobber = Field(dtype=bool, default=True,
                    doc="Clobber existing processed images?")
    isr = ConfigurableField(target=IsrTask, doc="ISR configuration")
    dateObs = Field(dtype=str, default="dateObs",
                    doc="Key for observation date in exposure registry")
    dateCalib = Field(dtype=str, default="calibDate",
                      doc="Key for calib date in calib registry")
    filter = Field(dtype=str, default="filter",
                   doc="Key for filter name in exposure/calib registries")
    combination = ConfigurableField(
        target=CalibCombineTask, doc="Calib combination configuration")
    ccdKeys = ListField(dtype=str, default=["ccd"],
                        doc="DataId keywords specifying a CCD")
    visitKeys = ListField(dtype=str, default=["visit"],
                          doc="DataId keywords specifying a visit")
    calibKeys = ListField(dtype=str, default=[],
                          doc="DataId keywords specifying a calibration")
    doCameraImage = Field(dtype=bool, default=True, doc="Create camera overview image?")
    binning = Field(dtype=int, default=64, doc="Binning to apply for camera image")

    def setDefaults(self):
        self.isr.doWrite = False


class CalibTaskRunner(TaskRunner):
    """Get parsed values into the CalibTask.run"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [dict(expRefList=parsedCmd.id.refList, butler=parsedCmd.butler, calibId=parsedCmd.calibId)]

    def __call__(self, args):
        """Call the Task with the kwargs from getTargetList"""
        task = self.TaskClass(config=self.config, log=self.log)
        if self.doRaise:
            result = task.run(**args)
        else:
            try:
                result = task.run(**args)
            except Exception as e:
                task.log.fatal("Failed: %s" % e)
                traceback.print_exc(file=sys.stderr)

        if self.doReturnResults:
            return Struct(
                args=args,
                metadata=task.metadata,
                result=result,
            )


class CalibTask(BatchPoolTask):
    """!Base class for constructing calibs.

    This should be subclassed for each of the required calib types.
    The subclass should be sure to define the following class variables:
    * _DefaultName: default name of the task, used by CmdLineTask
    * calibName: name of the calibration data set in the butler
    The subclass may optionally set:
    * filterName: filter name to give the resultant calib
    """
    ConfigClass = CalibConfig
    RunnerClass = CalibTaskRunner
    filterName = None
    calibName = None

    def __init__(self, *args, **kwargs):
        """Constructor"""
        BatchPoolTask.__init__(self, *args, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("combination")

    @classmethod
    def batchWallTime(cls, time, parsedCmd, numCores):
        numCcds = len(parsedCmd.butler.get("camera"))
        numExps = len(cls.RunnerClass.getTargetList(
            parsedCmd)[0]['expRefList'])
        numCycles = int(numCcds/float(numCores) + 0.5)
        return time*numExps*numCycles

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        return CalibArgumentParser(calibName=cls.calibName, name=cls._DefaultName, *args, **kwargs)

    def run(self, expRefList, butler, calibId):
        """!Construct a calib from a list of exposure references

        This is the entry point, called by the TaskRunner.__call__

        Only the master node executes this method.

        @param expRefList  List of data references at the exposure level
        @param butler      Data butler
        @param calibId   Identifier dict for calib
        """
        outputId = self.getOutputId(expRefList, calibId)
        ccdIdLists = getCcdIdListFromExposures(
            expRefList, level="sensor", ccdKeys=self.config.ccdKeys)

        # Ensure we can generate filenames for each output
        outputIdItemList = list(outputId.items())
        for ccdName in ccdIdLists:
            dataId = dict(outputIdItemList + [(k, ccdName[i])
                          for i, k in enumerate(self.config.ccdKeys)])
            try:
                butler.get(self.calibName + "_filename", dataId)
            except Exception as e:
                raise RuntimeError(
                    "Unable to determine output filename from %s: %s" % (dataId, e))

        processPool = Pool("process")
        processPool.storeSet(butler=butler)

        # Scatter: process CCDs independently
        data = self.scatterProcess(processPool, ccdIdLists)

        # Gather: determine scalings
        scales = self.scale(ccdIdLists, data)

        combinePool = Pool("combine")
        combinePool.storeSet(butler=butler)

        # Scatter: combine
        calibs = self.scatterCombine(combinePool, outputId, ccdIdLists, scales)

        if self.config.doCameraImage:
            try:
                self.makeCameraImage(butler, outputId, calibs)
            except Exception as exc:
                self.log.warn("Unable to create camera image: %s" % (exc,))

        return Struct(
            outputId = outputId,
            ccdIdLists = ccdIdLists,
            scales = scales,
            processPool = processPool,
            combinePool = combinePool,
            )

    def getOutputId(self, expRefList, calibId):
        """!Generate the data identifier for the output calib

        The mean date and the common filter are included, using keywords
        from the configuration.  The CCD-specific part is not included
        in the data identifier.

        @param expRefList  List of data references at exposure level
        @param calibId  Data identifier elements for the calib provided by the user
        @return data identifier
        """
        expIdList = [expRef.dataId for expRef in expRefList]
        midTime = 0
        filterName = None
        for expId in expIdList:
            midTime += self.getMjd(expId)
            thisFilter = self.getFilter(
                expId) if self.filterName is None else self.filterName
            if filterName is None:
                filterName = thisFilter
            elif filterName != thisFilter:
                raise RuntimeError("Filter mismatch for %s: %s vs %s" % (
                    expId, thisFilter, filterName))

        midTime /= len(expRefList)
        date = str(dafBase.DateTime(
            midTime, dafBase.DateTime.MJD).toPython().date())

        outputId = {self.config.filter: filterName,
                    self.config.dateCalib: date}
        outputId.update(calibId)
        return outputId

    def getMjd(self, dataId, timescale=dafBase.DateTime.UTC):
        """Determine the Modified Julian Date (MJD; in TAI) from a data identifier"""
        dateObs = dataId[self.config.dateObs]

        if "T" not in dateObs:
            dateObs = dateObs + "T12:00:00.0Z"
        elif not dateObs.endswith("Z"):
            dateObs += "Z"

        return dafBase.DateTime(dateObs, timescale).get(dafBase.DateTime.MJD)

    def getFilter(self, dataId):
        """Determine the filter from a data identifier"""
        return dataId[self.config.filter]

    def scatterProcess(self, pool, ccdIdLists):
        """!Scatter the processing among the nodes

        We scatter each CCD independently (exposures aren't grouped together),
        to make full use of all available processors. This necessitates piecing
        everything back together in the same format as ccdIdLists afterwards.

        Only the master node executes this method.

        @param pool  Process pool
        @param ccdIdLists  Dict of data identifier lists for each CCD name
        @return Dict of lists of returned data for each CCD name
        """
        self.log.info("Scatter processing")
        return mapToMatrix(pool, self.process, ccdIdLists)

    def process(self, cache, ccdId, outputName="postISRCCD", **kwargs):
        """!Process a CCD, specified by a data identifier

        After processing, optionally returns a result (produced by
        the 'processResult' method) calculated from the processed
        exposure.  These results will be gathered by the master node,
        and is a means for coordinated scaling of all CCDs for flats,
        etc.

        Only slave nodes execute this method.

        @param cache  Process pool cache
        @param ccdId  Data identifier for CCD
        @param outputName  Output dataset name for butler
        @return result from 'processResult'
        """
        if ccdId is None:
            self.log.warn("Null identifier received on %s" % NODE)
            return None
        sensorRef = getDataRef(cache.butler, ccdId)
        if self.config.clobber or not sensorRef.datasetExists(outputName):
            self.log.info("Processing %s on %s" % (ccdId, NODE))
            try:
                exposure = self.processSingle(sensorRef, **kwargs)
            except Exception as e:
                self.log.warn("Unable to process %s: %s" % (ccdId, e))
                raise
                return None
            self.processWrite(sensorRef, exposure)
        else:
            self.log.info(
                "Using previously persisted processed exposure for %s" % (sensorRef.dataId,))
            exposure = sensorRef.get(outputName, immediate=False)
        return self.processResult(exposure)

    def processSingle(self, dataRef):
        """Process a single CCD, specified by a data reference

        Generally, this simply means doing ISR.

        Only slave nodes execute this method.
        """
        return self.isr.runDataRef(dataRef).exposure

    def processWrite(self, dataRef, exposure, outputName="postISRCCD"):
        """!Write the processed CCD

        We need to write these out because we can't hold them all in
        memory at once.

        Only slave nodes execute this method.

        @param dataRef     Data reference
        @param exposure    CCD exposure to write
        @param outputName  Output dataset name for butler.
        """
        dataRef.put(exposure, outputName)

    def processResult(self, exposure):
        """Extract processing results from a processed exposure

        This method generates what is gathered by the master node.
        This can be a background measurement or similar for scaling
        flat-fields.  It must be picklable!

        Only slave nodes execute this method.
        """
        return None

    def scale(self, ccdIdLists, data):
        """!Determine scaling across CCDs and exposures

        This is necessary mainly for flats, so as to determine a
        consistent scaling across the entire focal plane.  This
        implementation is simply a placeholder.

        Only the master node executes this method.

        @param ccdIdLists  Dict of data identifier lists for each CCD tuple
        @param data        Dict of lists of returned data for each CCD tuple
        @return dict of Struct(ccdScale: scaling for CCD,
                               expScales: scaling for each exposure
                               ) for each CCD tuple
        """
        self.log.info("Scale on %s" % NODE)
        return dict((name, Struct(ccdScale=None, expScales=[None] * len(ccdIdLists[name])))
                    for name in ccdIdLists)

    def scatterCombine(self, pool, outputId, ccdIdLists, scales):
        """!Scatter the combination of exposures across multiple nodes

        In this case, we can only scatter across as many nodes as
        there are CCDs.

        Only the master node executes this method.

        @param pool  Process pool
        @param outputId  Output identifier (exposure part only)
        @param ccdIdLists  Dict of data identifier lists for each CCD name
        @param scales  Dict of structs with scales, for each CCD name
        @param dict of binned images
        """
        self.log.info("Scatter combination")
        outputIdItemList = outputId.items()
        data = [Struct(ccdIdList=ccdIdLists[ccdName], scales=scales[ccdName],
                       outputId=dict(outputIdItemList +
                                     [(k, ccdName[i]) for i, k in enumerate(self.config.ccdKeys)])) for
                ccdName in ccdIdLists]
        images = pool.map(self.combine, data)
        return dict(zip(ccdIdLists.keys(), images))

    def combine(self, cache, struct):
        """!Combine multiple exposures of a particular CCD and write the output

        Only the slave nodes execute this method.

        @param cache  Process pool cache
        @param struct  Parameters for the combination, which has the following components:
            * ccdIdList   List of data identifiers for combination
            * scales      Scales to apply (expScales are scalings for each exposure,
                               ccdScale is final scale for combined image)
            * outputId    Data identifier for combined image (fully qualified for this CCD)
        @return binned calib image
        """
        dataRefList = [getDataRef(cache.butler, dataId) if dataId is not None else None for
                       dataId in struct.ccdIdList]
        self.log.info("Combining %s on %s" % (struct.outputId, NODE))
        calib = self.combination.run(dataRefList, expScales=struct.scales.expScales,
                                     finalScale=struct.scales.ccdScale)

        self.recordCalibInputs(cache.butler, calib,
                               struct.ccdIdList, struct.outputId)

        self.interpolateNans(calib)

        self.write(cache.butler, calib, struct.outputId)

        return afwMath.binImage(calib.getImage(), self.config.binning)

    def recordCalibInputs(self, butler, calib, dataIdList, outputId):
        """!Record metadata including the inputs and creation details

        This metadata will go into the FITS header.

        @param butler  Data butler
        @param calib  Combined calib exposure.
        @param dataIdList  List of data identifiers for calibration inputs
        @param outputId  Data identifier for output
        """
        header = calib.getMetadata()
        header.add("OBSTYPE", self.calibName)  # Used by ingestCalibs.py

        # date, time, host, and root
        now = time.localtime()
        header.add("CALIB_CREATION_DATE", time.strftime("%Y-%m-%d", now))
        header.add("CALIB_CREATION_TIME", time.strftime("%X %Z", now))

        # Inputs
        visits = [str(dictToTuple(dataId, self.config.visitKeys)) for dataId in dataIdList if
                  dataId is not None]
        for i, v in enumerate(sorted(set(visits))):
            header.add("CALIB_INPUT_%d" % (i,), v)

        header.add("CALIB_ID", " ".join("%s=%s" % (key, value)
                                        for key, value in outputId.items()))
        checksum(calib, header)

    def interpolateNans(self, image):
        """Interpolate over NANs in the combined image

        NANs can result from masked areas on the CCD.  We don't want them getting
        into our science images, so we replace them with the median of the image.
        """
        if hasattr(image, "getMaskedImage"):  # Deal with Exposure vs Image
            self.interpolateNans(image.getMaskedImage().getVariance())
            image = image.getMaskedImage().getImage()
        if hasattr(image, "getImage"):  # Deal with DecoratedImage or MaskedImage vs Image
            image = image.getImage()
        array = image.getArray()
        bad = np.isnan(array)
        array[bad] = np.median(array[np.logical_not(bad)])

    def write(self, butler, exposure, dataId):
        """!Write the final combined calib

        Only the slave nodes execute this method

        @param butler  Data butler
        @param exposure  CCD exposure to write
        @param dataId  Data identifier for output
        """
        self.log.info("Writing %s on %s" % (dataId, NODE))
        butler.put(exposure, self.calibName, dataId)

    def makeCameraImage(self, butler, dataId, calibs):
        """!Create and write an image of the entire camera

        This is useful for judging the quality or getting an overview of
        the features of the calib.

        This requires that the 'ccd name' is a tuple containing only the
        detector ID.  If that is not the case, change CalibConfig.ccdKeys
        or set CalibConfig.doCameraImage=False to disable this.

        @param butler  Data butler
        @param dataId  Data identifier for output
        @param calibs  Dict mapping 'ccd name' to calib image
        """
        camera = butler.get("camera")

        class ImageSource(object):
            """Source of images for makeImageFromCamera

            This assumes that the 'ccd name' is a tuple containing
            only the detector ID.
            """
            def __init__(self, images):
                self.isTrimmed = True
                self.images = images
                self.background = np.nan

            def getCcdImage(self, detector, imageFactory, binSize):
                detId = (detector.getId(),)
                if detId not in self.images:
                    return imageFactory(1, 1)
                return self.images[detId]

        image = makeImageFromCamera(camera, imageSource=ImageSource(calibs), imageFactory=afwImage.ImageF,
                                    binSize=self.config.binning)
        butler.put(image, self.calibName + "_camera", dataId)


class BiasConfig(CalibConfig):
    """Configuration for bias construction.

    No changes required compared to the base class, but
    subclassed for distinction.
    """
    pass


class BiasTask(CalibTask):
    """Bias construction"""
    ConfigClass = BiasConfig
    _DefaultName = "bias"
    calibName = "bias"
    filterName = "NONE"  # Sets this filter name in the output

    @classmethod
    def applyOverrides(cls, config):
        """Overrides to apply for bias construction"""
        config.isr.doBias = False
        config.isr.doDark = False
        config.isr.doFlat = False
        config.isr.doFringe = False


class DarkCombineTask(CalibCombineTask):
    """Task to combine dark images"""
    def run(*args, **kwargs):
        combined = CalibCombineTask.run(*args, **kwargs)

        # Update the metadata
        visitInfo = afwImage.makeVisitInfo(exposureTime=1.0, darkTime=1.0)
        md = dafBase.PropertyList.cast(combined.getMetadata())
        afwImage.setVisitInfoMetadata(md, visitInfo)

        return combined


class DarkConfig(CalibConfig):
    """Configuration for dark construction"""
    doRepair = Field(dtype=bool, default=True, doc="Repair artifacts?")
    psfFwhm = Field(dtype=float, default=3.0, doc="Repair PSF FWHM (pixels)")
    psfSize = Field(dtype=int, default=21, doc="Repair PSF size (pixels)")
    crGrow = Field(dtype=int, default=2, doc="Grow radius for CR (pixels)")
    repair = ConfigurableField(
        target=RepairTask, doc="Task to repair artifacts")

    def setDefaults(self):
        CalibConfig.setDefaults(self)
        self.combination.retarget(DarkCombineTask)
        self.combination.mask.append("CR")


class DarkTask(CalibTask):
    """Dark construction

    The only major difference from the base class is a cosmic-ray
    identification stage, and dividing each image by the dark time
    to generate images of the dark rate.
    """
    ConfigClass = DarkConfig
    _DefaultName = "dark"
    calibName = "dark"
    filterName = "NONE"  # Sets this filter name in the output

    def __init__(self, *args, **kwargs):
        CalibTask.__init__(self, *args, **kwargs)
        self.makeSubtask("repair")

    @classmethod
    def applyOverrides(cls, config):
        """Overrides to apply for dark construction"""
        config.isr.doDark = False
        config.isr.doFlat = False
        config.isr.doFringe = False

    def processSingle(self, sensorRef):
        """Process a single CCD

        Besides the regular ISR, also masks cosmic-rays and divides each
        processed image by the dark time to generate images of the dark rate.
        The dark time is provided by the 'getDarkTime' method.
        """
        exposure = CalibTask.processSingle(self, sensorRef)

        if self.config.doRepair:
            psf = measAlg.DoubleGaussianPsf(self.config.psfSize, self.config.psfSize,
                                            self.config.psfFwhm/(2*math.sqrt(2*math.log(2))))
            exposure.setPsf(psf)
            self.repair.run(exposure, keepCRs=False)
            if self.config.crGrow > 0:
                mask = exposure.getMaskedImage().getMask().clone()
                mask &= mask.getPlaneBitMask("CR")
                fpSet = afwDet.FootprintSet(
                    mask.convertU(), afwDet.Threshold(0.5))
                fpSet = afwDet.FootprintSet(fpSet, self.config.crGrow, True)
                fpSet.setMask(exposure.getMaskedImage().getMask(), "CR")

        mi = exposure.getMaskedImage()
        mi /= self.getDarkTime(exposure)
        return exposure

    def getDarkTime(self, exposure):
        """Retrieve the dark time for an exposure"""
        darkTime = exposure.getInfo().getVisitInfo().getDarkTime()
        if not np.isfinite(darkTime):
            raise RuntimeError("Non-finite darkTime")
        return darkTime


class FlatConfig(CalibConfig):
    """Configuration for flat construction"""
    iterations = Field(dtype=int, default=10,
                       doc="Number of iterations for scale determination")
    stats = ConfigurableField(target=CalibStatsTask,
                              doc="Background statistics configuration")


class FlatTask(CalibTask):
    """Flat construction

    The principal change from the base class involves gathering the background
    values from each image and using them to determine the scalings for the final
    combination.
    """
    ConfigClass = FlatConfig
    _DefaultName = "flat"
    calibName = "flat"

    @classmethod
    def applyOverrides(cls, config):
        """Overrides for flat construction"""
        config.isr.doFlat = False
        config.isr.doFringe = False

    def __init__(self, *args, **kwargs):
        CalibTask.__init__(self, *args, **kwargs)
        self.makeSubtask("stats")

    def processResult(self, exposure):
        return self.stats.run(exposure)

    def scale(self, ccdIdLists, data):
        """Determine the scalings for the final combination

        We have a matrix B_ij = C_i E_j, where C_i is the relative scaling
        of one CCD to all the others in an exposure, and E_j is the scaling
        of the exposure.  We convert everything to logarithms so we can work
        with a linear system.  We determine the C_i and E_j from B_ij by iteration,
        under the additional constraint that the average CCD scale is unity.

        This algorithm comes from Eugene Magnier and Pan-STARRS.
        """
        assert len(ccdIdLists.values()) > 0, "No successful CCDs"
        lengths = set([len(expList) for expList in ccdIdLists.values()])
        assert len(
            lengths) == 1, "Number of successful exposures for each CCD differs"
        assert tuple(lengths)[0] > 0, "No successful exposures"
        # Format background measurements into a matrix
        indices = dict((name, i) for i, name in enumerate(ccdIdLists))
        bgMatrix = np.array([[0.0] * len(expList)
                            for expList in ccdIdLists.values()])
        for name in ccdIdLists:
            i = indices[name]
            bgMatrix[i] = [
                d if d is not None else np.nan for d in data[name]]

        numpyPrint = np.get_printoptions()
        np.set_printoptions(threshold='nan')
        self.log.info("Input backgrounds: %s" % bgMatrix)

        # Flat-field scaling
        numCcds = len(ccdIdLists)
        numExps = bgMatrix.shape[1]
        # log(Background) for each exposure/component
        bgMatrix = np.log(bgMatrix)
        bgMatrix = np.ma.masked_array(bgMatrix, np.isnan(bgMatrix))
        # Initial guess at log(scale) for each component
        compScales = np.zeros(numCcds)
        expScales = np.array(
            [(bgMatrix[:, i0] - compScales).mean() for i0 in range(numExps)])

        for iterate in range(self.config.iterations):
            compScales = np.array(
                [(bgMatrix[i1, :] - expScales).mean() for i1 in range(numCcds)])
            expScales = np.array(
                [(bgMatrix[:, i2] - compScales).mean() for i2 in range(numExps)])

            avgScale = np.average(np.exp(compScales))
            compScales -= np.log(avgScale)
            self.log.debug("Iteration %d exposure scales: %s",
                           iterate, np.exp(expScales))
            self.log.debug("Iteration %d component scales: %s",
                           iterate, np.exp(compScales))

        expScales = np.array(
            [(bgMatrix[:, i3] - compScales).mean() for i3 in range(numExps)])

        if np.any(np.isnan(expScales)):
            raise RuntimeError("Bad exposure scales: %s --> %s" %
                               (bgMatrix, expScales))

        expScales = np.exp(expScales)
        compScales = np.exp(compScales)

        self.log.info("Exposure scales: %s" % expScales)
        self.log.info("Component relative scaling: %s" % compScales)
        np.set_printoptions(**numpyPrint)

        return dict((ccdName, Struct(ccdScale=compScales[indices[ccdName]], expScales=expScales))
                    for ccdName in ccdIdLists)


class FringeConfig(CalibConfig):
    """Configuration for fringe construction"""
    stats = ConfigurableField(target=CalibStatsTask,
                              doc="Background statistics configuration")
    subtractBackground = ConfigurableField(target=measAlg.SubtractBackgroundTask,
                                           doc="Background configuration")
    detection = ConfigurableField(
        target=measAlg.SourceDetectionTask, doc="Detection configuration")
    detectSigma = Field(dtype=float, default=1.0,
                        doc="Detection PSF gaussian sigma")


class FringeTask(CalibTask):
    """Fringe construction task

    The principal change from the base class is that the images are
    background-subtracted and rescaled by the background.

    XXX This is probably not right for a straight-up combination, as we
    are currently doing, since the fringe amplitudes need not scale with
    the continuum.

    XXX Would like to have this do PCA and generate multiple images, but
    that will take a bit of work with the persistence code.
    """
    ConfigClass = FringeConfig
    _DefaultName = "fringe"
    calibName = "fringe"

    @classmethod
    def applyOverrides(cls, config):
        """Overrides for fringe construction"""
        config.isr.doFringe = False

    def __init__(self, *args, **kwargs):
        CalibTask.__init__(self, *args, **kwargs)
        self.makeSubtask("detection")
        self.makeSubtask("stats")
        self.makeSubtask("subtractBackground")

    def processSingle(self, sensorRef):
        """Subtract the background and normalise by the background level"""
        exposure = CalibTask.processSingle(self, sensorRef)
        bgLevel = self.stats.run(exposure)
        self.subtractBackground.run(exposure)
        mi = exposure.getMaskedImage()
        mi /= bgLevel
        footprintSets = self.detection.detectFootprints(
            exposure, sigma=self.config.detectSigma)
        mask = exposure.getMaskedImage().getMask()
        detected = 1 << mask.addMaskPlane("DETECTED")
        for fpSet in (footprintSets.positive, footprintSets.negative):
            if fpSet is not None:
                afwDet.setMaskFromFootprintList(
                    mask, fpSet.getFootprints(), detected)
        return exposure


def pca(data, numComponents=None):
    """Principal Components Analysis

    From: http://stackoverflow.com/a/13224592/834250

    @param data  numpy array of data to analyse
    @param numComponents  number of principal components to use
    @return principal components as numpy array, eigenvalues, eigenvectors
    """
    m, n = data.shape
    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=False)
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    if numComponents is not None:
        evecs = evecs[:, :numComponents]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T, evals, evecs


class RPCA(object):
    """Robust PCA

    Models data matrix D as L + S, where L is a low-rank matrix
    (contains common elements) and S is a sparse matrix (contains
    parts particular to individual elements).

    Usage:

    >>> L, S = RPCA(D).fit()

    From https://github.com/dganguli/robust-pca/blob/master/r_pca.py
    """
    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.norm_p(self.D, 2))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def norm_p(M, p):
        return np.sum(np.power(M, p))

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iteration = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-9 * self.norm_p(np.abs(self.D), 2)

        while (err > _tol) and iteration < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(np.abs(self.D - Lk - Sk), 2)
            iteration += 1

        self.L = Lk
        self.S = Sk
        return Lk, Sk


class BaseSkyConfig(CalibConfig):
    """Base configuration for sky model construction"""
    detection = ConfigurableField(target=measAlg.SourceDetectionTask, doc="Detection configuration")
    detectSigma = Field(dtype=float, default=2.0, doc="Detection PSF gaussian sigma")
    subtractBackground = ConfigurableField(target=measAlg.SubtractBackgroundTask,
                                           doc="Background configuration")
    sky = ConfigurableField(target=SkyMeasurementTask, doc="Sky measurement")
    maskThresh = Field(dtype=float, default=3.0, doc="k-sigma threshold for masking pixels")
    mask = ListField(dtype=str, default=["BAD", "SAT", "DETECTED", "NO_DATA"],
                     doc="Mask planes to consider as contaminated")


class BaseSkyTask(CalibTask):
    """Task for sky model construction"""
    ConfigClass = BaseSkyConfig

    def __init__(self, *args, **kwargs):
        CalibTask.__init__(self, *args, **kwargs)
        self.makeSubtask("detection")
        self.makeSubtask("subtractBackground")
        self.makeSubtask("sky")

    def _getConfigName(self):
        return None
    def _getMetadataName(self):
        return None

    def scatterProcess(self, pool, ccdIdLists):
        """!Scatter the processing among the nodes

        Only the master node executes this method, assigning work to the
        slaves.

        We subtract off a large-scale background model across all CCDs,
        which requires a scatter/gather. Then we process the individual
        CCDs, subtracting the large-scale background model and the
        residual background model measured. These residuals will be
        combined for the sky frame.

        @param pool  Process pool
        @param ccdIdLists  Dict of data identifier lists for each CCD name
        @return Dict of lists of returned data for each CCD name
        """
        self.log.info("Scatter processing")

        numExps = set(len(expList) for expList in ccdIdLists.values())
        assert len(numExps) == 1
        numExps = numExps.pop()

        # First subtract off general gradients to make all the exposures look similar.
        # We want to preserve the common small-scale structure, which we will coadd.
        bgModelList = mapToMatrix(pool, self.measureBackground, ccdIdLists)

        backgrounds = {}
        scales = {}
        for exp in range(numExps):
            bgModels = [bgModelList[ccdName][exp] for ccdName in ccdIdLists]
            visit = set(tuple(ccdIdLists[ccdName][exp][key] for key in sorted(self.config.visitKeys)) for
                        ccdName in ccdIdLists)
            assert len(visit) == 1
            visit = visit.pop()
            bgModel = bgModels[0]
            for bg in bgModels[1:]:
                bgModel.merge(bg)
            backgrounds[visit] = bgModel
            scales[visit] = np.median(bgModel.getStatsImage().getArray())

        return mapToMatrix(pool, self.process, ccdIdLists, backgrounds=backgrounds, scales=scales)

    def measureBackground(self, cache, dataId):
        """!Measure background model for CCD

        This method is executed by the slaves.

        The background models for all CCDs in an exposure will be
        combined to form a full focal-plane background model.

        @param cache  Process pool cache
        @param dataId  Data identifier
        @return Bcakground model
        """
        dataRef = getDataRef(cache.butler, dataId)
        exposure = self.processSingleBackground(dataRef)

        # NAOJ prototype smoothed and then combined the entire image, but it shouldn't be any different
        # to bin and combine the binned images except that there's fewer pixels to worry about.
        config = FocalPlaneBackgroundConfig()
        camera = dataRef.get("camera")
        bgModel = FocalPlaneBackground.fromCamera(config, camera)
        bgModel.addCcd(exposure)
        return bgModel

    def processSingleBackground(self, dataRef):
        """!Process a single CCD for the background

        This method is executed by the slaves.

        Because we're interested in the background, we detect and mask astrophysical
        sources, and pixels above the noise level.

        @param dataRef  Data reference for CCD.
        @return processed exposure
        """
        if not self.config.clobber and dataRef.datasetExists("postISRCCD"):
            return dataRef.get("postISRCCD")
        exposure = CalibTask.processSingle(self, dataRef)

        # Detect sources. Requires us to remove the background; we'll restore it later.
        bgTemp = self.subtractBackground.run(exposure).background
        self.detection.detectFootprints(exposure, sigma=self.config.detectSigma)
        image = exposure.getMaskedImage()

        # Mask high pixels
        variance = image.getVariance()
        noise = np.sqrt(np.median(variance.getArray()))
        isHigh = image.getImage().getArray() > self.config.maskThresh*noise
        image.getMask().getArray()[isHigh] |= image.getMask().getPlaneBitMask("DETECTED")

        # Restore the background: it's what we want!
        image += bgTemp.getImage()

        # Set detected/bad pixels to background to ensure they don't corrupt the background
        maskVal = image.getMask().getPlaneBitMask(self.config.mask)
        isBad = image.getMask().getArray() & maskVal > 0
        bgLevel = np.median(image.getImage().getArray()[~isBad])
        image.getImage().getArray()[isBad] = bgLevel
        dataRef.put(exposure, "postISRCCD")
        return exposure

    def processSingle(self, dataRef, backgrounds, scales):
        """Process a single CCD, specified by a data reference

        We subtract the appropriate focal plane background model,
        divide by the appropriate scale and measure the background.

        Only slave nodes execute this method.

        @param dataRef  Data reference for single CCD
        @param backgrounds  Background model for each visit
        @param scales  Scales for each visit
        @return Processed exposure
        """
        visit = tuple(dataRef.dataId[key] for key in sorted(self.config.visitKeys))
        exposure = dataRef.get("postISRCCD", immediate=True)
        image = exposure.getMaskedImage()
        detector = exposure.getDetector()
        bbox = image.getBBox()

        bgModel = backgrounds[visit]
        bg = bgModel.toCcdBackground(detector, bbox)
        image -= bg.getImage()
        image /= scales[visit]

        bg = self.sky.measureBackground(exposure.getMaskedImage())
        dataRef.put(bg, "icExpBackground")
        return exposure


class PrimarySkyConfig(BaseSkyConfig):
    def setDefaults(self):
        self.sky.background.xBinSize = 64
        self.sky.background.yBinSize = 64


class PrimarySkyTask(BaseSkyTask):
    """Primary sky frame construction

    The primary sky frame is a (relatively) small-scale background
    model, the response of the camera to the sky.

    We might also construct secondary sky frames, which will be
    larger-scale corrections.
    """
    ConfigClass = PrimarySkyConfig
    _DefaultName = "sky"
    calibName = "sky"

    def combine(self, cache, struct):
        """!Combine multiple background models of a particular CCD and write the output

        Only the slave nodes execute this method.

        @param cache  Process pool cache
        @param struct  Parameters for the combination, which has the following components:
            * ccdIdList   List of data identifiers for combination
            * outputId    Data identifier for combined image (fully qualified for this CCD)
        @return binned calib image
        """
        dataRefList = [getDataRef(cache.butler, dataId) if dataId is not None else None for
                       dataId in struct.ccdIdList]
        self.log.info("Combining %s on %s" % (struct.outputId, NODE))
        bgList = [dataRef.get("icExpBackground", immediate=True).clone() for dataRef in dataRefList]

        bgExp = self.sky.averageBackgrounds(bgList)

        self.recordCalibInputs(cache.butler, bgExp, struct.ccdIdList, struct.outputId)
        cache.butler.put(bgExp, "sky", struct.outputId)
        return bgExp.getMaskedImage().getImage()


class SecondarySkyConfig(BaseSkyConfig):
    numComponents = Field(dtype=int, default=10, doc="Number of principal components")

    def setDefaults(self):
        self.sky.background.xBinSize = 512
        self.sky.background.yBinSize = 1000


class SecondarySkyTask(BaseSkyTask):
    """Secondary sky frame construction

    The secondary sky frame is a collection of large-scale background
    models, the residuals after subtraction of the primary (small-scale)
    sky frame.
    """
    ConfigClass = SecondarySkyConfig
    _DefaultName = "sky2"
    calibName = "sky2"

    def scatterCombine(self, pool, outputId, ccdIdLists, scales):
        """!Scatter the combination of exposures across multiple nodes

        Only the master node executes this method.

        We first measure the scales for the primary sky frame, apply
        that and collect all the background models for principal
        component analysis.  The principal components form the secondary
        sky frame.

        @param pool  Process pool
        @param outputId  Output identifier (exposure part only)
        @param ccdIdLists  Dict of data identifier lists for each CCD name
        @param scales  Dict of structs with scales, for each CCD name
        @return dict of binned images
        """
        self.log.info("Scatter collection")
        numExps = set(len(expList) for expList in ccdIdLists.values())
        assert len(numExps) == 1
        numExps = numExps.pop()
        numCcds = len(ccdIdLists)

        visits = [set(tuple(ccdIdLists[ccdName][exp][key] for key in self.config.visitKeys) for
                      ccdName in ccdIdLists) for exp in range(numExps)]
        assert all(len(vv) == 1 for vv in visits)
        visits = [vv.pop() for vv in visits]

        measPrimary = mapToMatrix(pool, self.measurePrimary, ccdIdLists)
        primaries = {visits[exp]: self.sky.solveScales([measPrimary[ccdName][exp]
                                                      for ccdName in ccdIdLists])
                     for exp in range(numExps)}
        self.log.info("Primary scales: %s" % (primaries,))

        # XXX possible improvement: iteratively combine frames rather than PCA
        bgMatrix = mapToMatrix(pool, self.collectCombine, ccdIdLists, primaries)
        boxes = [[bgMatrix[ccdName][exp].box for exp in range(numExps)] for ccdName in ccdIdLists]
        dims = [set((boxes[ccd][exp].getWidth(), boxes[ccd][exp].getHeight()) for
                     exp in range(numExps)) for ccd in range(numCcds)]
        assert all(len(dd) == 1 for dd in dims)
        arrayMatrix = [[bgMatrix[ccdName][exp].statsImage.getImage().getArray() for exp in range(numExps)] for
                       ccdName in ccdIdLists]

        shapes = [set(arrayMatrix[ccd][exp].shape for exp in range(numExps)) for ccd in range(numCcds)]
        assert all(len(ss) == 1 for ss in shapes), "Differing shapes for CCD: %s" % (shapes,)
        shapes = [ss.pop() for ss in shapes]
        numPix = [ss[0]*ss[1] for ss in shapes]

        # Join all CCDs within an exposure
        exposures = [np.concatenate([arrayMatrix[ccd][exp].reshape((numPix[ccd], 1)) for
                                        ccd in range(numCcds)], axis=0) for exp in range(numExps)]
        data = np.concatenate(exposures, axis=1)

        # Replace bad pixels with the mean
        isBad = ~np.isfinite(data)
        numPixels = np.ones_like(data, dtype=int)
        numPixels[isBad] = 0
        data[isBad] = 0.0
        mean = np.sum(data, axis=1)/np.sum(numPixels, axis=1)
        mean[~np.isfinite(mean)] = 0.0
        for ii in range(numExps):
            if isBad[ii].sum() > 0:
                data[ii][isBad[ii]] = mean[isBad[ii]]

        self.log.info("Performing PCA")
        numComponents = min(numExps, self.config.numComponents)
        components, evals, evecs = pca(data) # , numComponents)
        self.log.info("Normalised eigenvalues: %s" % (evals/np.sum(evals),))

        self.log.info("Performing robust PCA")
        L, S = RPCA(data).fit()
        rcomp, revals, revecs = pca(L) # , numComponents)
        self.log.info("Normalised robust eigenvalues: %s" % (revals/np.sum(revals),))
        components, evals, evecs = rcomp, revals, revecs

        # Normalise each component: normalisation isn't important because we'll fit coefficients
        for ii in range(numComponents):
            norm = np.median(components.T[ii])
            components.T[ii] /= norm

        # Pull the CCDs out of the exposure array
        offsets = [sum(numPix[:ccd]) for ccd in range(numCcds)]
        componentsByCcd = [[components.T[comp][offsets[ccd]:offsets[ccd] + numPix[ccd]].reshape(shapes[ccd])
                            for comp in range(numComponents)] for ccd in range(numCcds)]

        writeData = [Struct(components=componentsByCcd[ccd],
                            dims=dims[ccd].pop(),
                            outputId=dict(outputId.items() + [(k, ccdName[i]) for
                                          i, k in enumerate(self.config.ccdKeys)]),
                            dataIdList=ccdIdLists[ccdName],
                            ) for ccd, ccdName in enumerate(ccdIdLists)]
        images = pool.map(self.writeComponents, writeData)
        return dict(zip(ccdIdLists.keys(), images))

    def measurePrimary(self, cache, dataId):
        """Measure the scale for the primary sky frame

        This method is executed on the slaves.

        @param cache  Process pool cache
        @param dataId  Data identifier for CCD
        @return measured scale
        """
        exposure = cache.butler.get("postISRCCD", dataId, immediate=True)
        sky = self.sky.getSkyData(cache.butler, dataId)
        return self.sky.measureScale(exposure.getMaskedImage(), sky)

    def collectCombine(self, cache, dataId, primaries):
        """Collect background models for combination

        We subtract the primary sky frame and measure the background.

        This method is executed on the slaves.

        @param cache  Process pool cache
        @param dataId  Data identifier for CCD
        @param primaries  Scale factors for primary sky frame
        @return Struct(statsImage: binned image, box: image bounding box)
        """
        exposure = cache.butler.get("postISRCCD", dataId, immediate=True)
        sky = self.sky.getSkyData(cache.butler, dataId)
        visit = tuple(dataId[key] for key in self.config.visitKeys)
        self.sky.subtractSkyFrame(exposure.getMaskedImage(), sky, primaries[visit])

        bg = self.sky.measureBackground(exposure.getMaskedImage())
        assert len(bg) == 1
        return Struct(statsImage=bg[0][0].getStatsImage(), box=bg[0][0].getImageBBox())

    def writeComponents(self, cache, struct):
        """Write the secondary sky frame components for a CCD

        This method is executed on the slaves.

        @param cache  Process pool cache
        @param struct  Struct(dataIdList: list of data identifiers,
                              outputId: output data identifier,
                              dims: dimensions of image)
        @return First component image, for display purposes
        """
        width, height = struct.dims
        box = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(width, height))
        calib = self.sky.componentsToImage(struct.components, box)
        self.recordCalibInputs(cache.butler, calib, struct.dataIdList, struct.outputId)
        cache.butler.put(calib, self.calibName, struct.outputId)
        # Return the first only for display purposes
        return afwMath.binImage(self.sky.imageToComponents(calib)[0].getImage(), self.config.binning)
