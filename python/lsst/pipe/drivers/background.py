import numpy
import itertools

import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as afwCameraGeom

from lsst.pex.config import Config, Field, ListField, ChoiceField, ConfigField, RangeField
from lsst.pipe.base import Task


def robustMean(array, rej=3.0):
    """Measure a robust mean of an array

    Parameters
    ----------
    array : `numpy.ndarray`
        Array for which to measure the mean.
    rej : `float`
        k-sigma rejection threshold.

    Returns
    -------
    mean : `array.dtype`
        Robust mean of `array`.
    """
    q1, median, q3 = numpy.percentile(array, [25.0, 50.0, 100.0])
    good = numpy.abs(array - median) < rej*0.74*(q3 - q1)
    return array[good].mean()


class BackgroundConfig(Config):
    """Configuration for background measurement"""
    statistic = ChoiceField(
        doc="type of statistic to use for grid points",
        dtype=str, default="MEANCLIP",
        allowed={
            "MEANCLIP": "clipped mean",
            "MEAN": "unclipped mean",
            "MEDIAN": "median",
        }
    )
    xBinSize = RangeField(
        doc="how large a region of the sky should be used for each background point",
        dtype=int, default=512, min=1,
    )
    yBinSize = RangeField(
        doc="how large a region of the sky should be used for each background point",
        dtype=int, default=512, min=1,
    )
    algorithm = ChoiceField(
        doc="how to interpolate the background values. This maps to an enum; see afw::math::Background",
        dtype=str, default="NATURAL_SPLINE", optional=True,
        allowed={
            "CONSTANT": "Use a single constant value",
            "LINEAR": "Use linear interpolation",
            "NATURAL_SPLINE": "cubic spline with zero second derivative at endpoints",
            "AKIMA_SPLINE": "higher-level nonlinear spline that is more robust to outliers",
            "NONE": "No background estimation is to be attempted",
        },
    )
    mask = ListField(
        doc="Names of mask planes to ignore while estimating the background",
        dtype=str, default=["SAT", "BAD", "EDGE", "DETECTED", "DETECTED_NEGATIVE", "NO_DATA", ],
    )


class SkyStatsConfig(Config):
    """Parameters controlling the measurement of sky statistics"""
    statistic = ChoiceField(
        doc="statistic to use",
        dtype=str, default="MEANCLIP",
        allowed={
            "MEANCLIP": "clipped mean",
            "MEAN": "unclipped mean",
            "MEDIAN": "median",
        }
    )
    clip = Field(doc="Clipping threshold for background", dtype=float, default=3.0)
    nIter = Field(doc="Clipping iterations for background", dtype=int, default=3)
    mask = ListField(doc="Mask planes to reject", dtype=str, default=["SAT", "DETECTED", "BAD", "NO_DATA",])


class SkyMeasurementConfig(Config):
    """Configuration for SkyMeasurementTask"""
    stats = ConfigField(dtype=SkyStatsConfig, doc="Measurement of sky statistics")
    skyRej = Field(dtype=float, default=3.0, doc="k-sigma rejection threshold for sky scale")
    pistonRej = Field(dtype=float, default=3.0, doc="k-sigma rejection threshold for pattern scale")
    background = ConfigField(dtype=BackgroundConfig, doc="Background measurement")


class SkyMeasurementTask(Task):
    """Task for creating, persisting and using sky frames

    A sky frame is like a fringe frame (the sum of many exposures of the night sky,
    combined with rejection to remove astrophysical objects) except the structure
    is on larger scales, and hence we bin the images and represent them as a
    background model (a `lsst.afw.math.BackgroundMI`).  The sky frame represents
    the dominant response of the camera to the sky background.

    Higher-order terms ("components") may be present, e.g., due to the spatial
    variation of color terms over the camera.  Though there is some code here to
    support those, it isn't yet working ideally and so has been disabled.
    """
    ConfigClass = SkyMeasurementConfig

    # def putSkyData(self, butler, calibId, bgExp, pistons=None):
    #     self.addPistonHeaders(bgExp, pistons)
    #     butler.put(bgExp, "sky", calibId)

    def getSkyData(self, butler, calibId):
        """Retrieve sky frame from the butler

        Parameters
        ----------
        butler : `lsst.daf.persistence.Butler`
            Data butler
        calibId : `dict`
            Data identifier for calib

        Returns
        -------
        sky : `lsst.afw.math.BackgroundList`
            Sky frame
        """
        exp = butler.get("sky", calibId, immediate=True)
        return self.exposureToBackground(exp)

    @staticmethod
    def exposureToBackground(bgExp):
        """Convert an exposure to background model

        Calibs need to be persisted as an Exposure, so we need to convert
        the persisted Exposure to a background model.

        Parameters
        ----------
        bgExp : `lsst.afw.image.Exposure`
            Background model in Exposure format.

        Returns
        -------
        bg : `lsst.afw.math.BackgroundList`
            Background model
        """
        header = bgExp.getMetadata()
        xMin = header.get("BOX.MINX")
        yMin = header.get("BOX.MINY")
        xMax = header.get("BOX.MAXX")
        yMax = header.get("BOX.MAXY")
        algorithm = header.get("ALGORITHM")
        bbox = afwGeom.Box2I(afwGeom.Point2I(xMin, yMin), afwGeom.Point2I(xMax, yMax))
        return afwMath.BackgroundList(
                (afwMath.BackgroundMI(bbox, bgExp.getMaskedImage()),
                 afwMath.stringToInterpStyle(algorithm),
                 afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
                 afwMath.ApproximateControl.UNKNOWN,
                 0, 0, False))

    def backgroundToExposure(self, statsImage, bbox):
        """Convert a background model to an exposure

        Calibs need to be persisted as an Exposure, so we need to convert
        the background model to an Exposure

        Parameters
        ----------
        statsImage : `lsst.afw.image.MaskedImageF`
            Background model's statistics image.
        bbox : `lsst.afw.geom.Box2I`
            Bounding box for image.

        Returns
        -------
        exp : `lsst.afw.image.Exposure`
            Background model in Exposure format.
        """
        exp = afwImage.makeExposure(statsImage)
        header = exp.getMetadata()
        header.set("BOX.MINX", bbox.getMinX())
        header.set("BOX.MINY", bbox.getMinY())
        header.set("BOX.MAXX", bbox.getMaxX())
        header.set("BOX.MAXY", bbox.getMaxY())
        header.set("ALGORITHM", self.config.background.algorithm)
        return exp

    def measureBackground(self, image):
        """Measure a background model for image

        This doesn't use a full-featured background model (e.g., no Chebyshev
        approximation) because we just want the binning behaviour.  This will
        allow us to average the bins later (`averageBackgrounds`).

        The `BackgroundMI` is wrapped in a `BackgroundList` so it can be
        pickled and persisted.

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Image for which to measure background.

        Returns
        -------
        bgModel : `lsst.afw.math.BackgroundList`
            Background model.
        """
        stats = afwMath.StatisticsControl()
        stats.setAndMask(image.getMask().getPlaneBitMask(self.config.background.mask))
        stats.setNanSafe(True)
        ctrl = afwMath.BackgroundControl(
            self.config.background.algorithm,
            max(int(image.getWidth()/self.config.background.xBinSize + 0.5), 1),
            max(int(image.getHeight()/self.config.background.yBinSize + 0.5), 1),
            "REDUCE_INTERP_ORDER",
            stats,
            self.config.background.statistic
        )

        bg = afwMath.makeBackground(image, ctrl)

        return afwMath.BackgroundList((
            afwMath.cast_BackgroundMI(bg),
            self.config.background.algorithm,
            afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
            afwMath.ApproximateControl.UNKNOWN,
            0, 0, False
        ))

    def averageBackgrounds(self, bgList):
        """Average multiple background models

        The input background models should be a `BackgroundList` consisting
        of a single `BackgroundMI`.

        Parameters
        ----------
        bgList : `list` of `lsst.afw.math.BackgroundList`
            Background models to average.

        Returns
        -------
        bgExp : `lsst.afw.image.Exposure`
            Background model in Exposure format.
        """
        assert all(len(bg) == 1 for bg in bgList), "Mixed bgList: %s" % ([len(bg) for bg in bgList],)
        images = [bg[0][0].getStatsImage() for bg in bgList]
        boxes = [bg[0][0].getImageBBox() for bg in bgList]
        assert len(set((box.getMinX(), box.getMinY(), box.getMaxX(), box.getMaxY()) for box in boxes)) == 1
        bbox = boxes.pop(0)

        # Ensure bad pixels are masked
        maskVal = afwImage.MaskU.getPlaneBitMask("BAD")
        for img in images:
            bad = numpy.isnan(img.getImage().getArray())
            img.getMask().getArray()[bad] = maskVal

        stats = afwMath.StatisticsControl()
        stats.setAndMask(maskVal)
        stats.setNanSafe(True)
        combined = afwMath.statisticsStack(afwImage.vectorMaskedImageF(images), afwMath.MEANCLIP, stats)

        # Set bad pixels
        array = combined.getImage().getArray()
        bad = numpy.isnan(array)
        mean = robustMean(array[~bad], self.config.skyRej)
        array[bad] = mean

        # Put it into an exposure, which is required for calibs
        return self.backgroundToExposure(combined, bbox)

    def measureScale(self, image, skyBackground):
        """Measure scale of background model in image

        Apart from astrophysical sources and bad pixels, the image and
        background model should differ only by a scale factor.

        Parameters
        ----------
        image : `lsst.afw.image.Exposure` or `lsst.afw.image.MaskedImage`
            Science image for which to measure scale.
        skyBackground : `lsst.afw.math.BackgroundList`
            Sky background model.

        Returns
        -------
        scale : `float`
            Scale factor.
        """
        if hasattr(image, "getMaskedImage"):
            image = image.getMaskedImage()
        if True:
            image = image.clone()
            image /= skyBackground.getImage()
            maskVal = image.getMask().getPlaneBitMask(self.config.stats.mask)
            ctrl = afwMath.StatisticsControl(self.config.stats.clip, self.config.stats.nIter, maskVal)
            statistic = afwMath.stringToStatisticsProperty(self.config.stats.statistic)
            stats = afwMath.makeStatistics(image, statistic, ctrl)
            return stats.getValue(statistic)
        else:
            # No rejection: less robust
            maskVal = image.getMask().getPlaneBitMask(self.config.stats.mask)
            isGood = image.getMask().getArray() & maskVal == 0
            array = image.getImage().getArray()[isGood]
            skyImage = skyBackground.getImage()
            skyArray = skyImage.getArray()[isGood]
            dataDotModel = numpy.dot(array, skyArray)
            modelDotModel = numpy.dot(skyArray, skyArray)
            return dataDotModel/modelDotModel

    def solveScales(self, scales):
        """Solve multiple scales for a single scale factor

        We have multiple scales from the different CCDs, and want
        a single scale factor for the entire exposure.

        Parameters
        ----------
        scales : `numpy.ndarray` or `list` of `float`
            Scale factors.

        Returns
        -------
        meanScale : `float`
            Mean scale factor.
        """
        return robustMean(numpy.array(scales), self.config.skyRej)

    def subtractSkyFrame(self, image, skyBackground, scale, bgList=None):
        """Subtract sky frame from science image

        Parameters
        ----------
        image : `lsst.afw.image.Exposure` or `lsst.afw.image.MaskedImage`
            Science image.
        skyBackground : `lsst.afw.math.BackgroundList`
            Sky background model.
        scale : `float`
            Scale to apply to background model.
        bgList : `lsst.afw.math.BackgroundList`
            List of backgrounds applied to image
        """
        if hasattr(image, "getMaskedImage"):
            image = image.getMaskedImage()
        if hasattr(image, "getImage"):
            image = image.getImage()
        image.scaledMinus(scale, skyBackground.getImage())
        if bgList is not None:
            bgData = list(skyBackground[0])
            bg = bgData[0]
            statsImage = bg.getStatsImage().clone()
            statsImage *= scale
            newBg = afwMath.BackgroundMI(bg.getImageBBox(), statsImage)
            newBgData = [newBg] + bgData[1:]
            bgList.append(newBgData)

    def componentsToImage(self, components, imageBox):
        """Convert sky components to an image

        When the sky model consists of multiple components (e.g., from a PCA),
        we don't have the mechanisms to read/write an image cube. Instead,
        we'll stitch the multiple images together into one big image (separated
        by a column of -INF so the division is clear).

        Parameters
        ----------
        components : `list` of `numpy.ndarray`
            Components of sky model.
        imageBox : `lsst.afw.geom.Box2I`
            Bounding box for image.

        Returns
        -------
        compImage : `lsst.afw.image.DecoratedImageF`
            Image with sky components.
        """
        shapes = set(arr.shape for arr in components)
        assert len(shapes) == 1
        height, width = shapes.pop()  # numpy shape is h,w

        num = len(components)

        combined = afwImage.ImageF((width + 1)*num, height)
        array = combined.getArray()
        for ii in range(num):
            left = ii*(width + 1)
            right = left + width
            array[:, left:right] = components[ii].reshape((height, width))
            array[:, right] = -numpy.inf

        image = afwImage.DecoratedImageF(combined)
        header = image.getMetadata()
        header.add("NUM.COMPONENTS", num)
        header.add("COMPONENT.WIDTH", width)
        header.add("COMPONENT.HEIGHT", height)
        header.set("BOX.MINX", imageBox.getMinX())
        header.set("BOX.MINY", imageBox.getMinY())
        header.set("BOX.MAXX", imageBox.getMaxX())
        header.set("BOX.MAXY", imageBox.getMaxY())
        header.set("ALGORITHM", self.config.background.algorithm)
        return image

    def imageToComponents(self, compImage):
        """Convert image to sky components

        When the sky model consists of multiple components (e.g., from a PCA),
        we don't have the mechanisms to read/write an image cube. Instead,
        we'll stitch the multiple images together into one big image (separated
        by a column of -INF so the division is clear).

        Parameters
        ----------
        compImage : `lsst.afw.image.DecoratedImageF`
            Image with sky components.

        Returns
        -------
        bgModels : `list` of `lsst.afw.math.BackgroundList`
            Components of sky model.
        """
        header = compImage.getMetadata()
        num = header.get("NUM.COMPONENTS")
        width = header.get("COMPONENT.WIDTH")
        height = header.get("COMPONENT.HEIGHT")
        xMin = header.get("BOX.MINX")
        yMin = header.get("BOX.MINY")
        xMax = header.get("BOX.MAXX")
        yMax = header.get("BOX.MAXY")
        algorithm = header.get("ALGORITHM")
        assert compImage.getImage().getHeight() == height
        assert compImage.getImage().getWidth() == num*(width + 1)
        components = [compImage.getImage().getArray()[:, ii*(width + 1):(ii + 1)*(width + 1) - 1] for
                      ii in range(num)]
        bbox = afwGeom.Box2I(afwGeom.Point2I(xMin, yMin), afwGeom.Point2I(xMax, yMax))
        bgImages = [afwImage.MaskedImageF(width, height) for ii in range(num)]
        for img, comp in zip(bgImages, components):
            img.getImage().getArray()[:] = comp
        bgModels = [afwMath.BackgroundList(
                (afwMath.BackgroundMI(bbox, img),
                 afwMath.stringToInterpStyle(algorithm),
                 afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
                 afwMath.ApproximateControl.UNKNOWN,
                 0, 0, False)) for img in bgImages]
        return bgModels

    def measureComponents(self, image, bgModels):
        """Measure background model components

        We construct the least-squares equation for fitting a linear
        combination of background models to the science image.

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Science image to measure.
        bgModels : `list` of `lsst.afw.math.BackgroundList`
            Background model components.

        Returns
        -------
        matrix : `numpy.ndarray`
            Least-squares matrix ("Fisher matrix").
        vector : `numpy.ndarray`
            Least-squares vector ("right-hand side").
        """
        num = len(bgModels)

        mask = image.getMask()
        maskVal = mask.getPlaneBitMask(self.config.background.mask)
        isGood = mask & maskVal == 0
        array = image.getImage().getArray()[isGood]

        # Need matrix from model.dot.model, vector from data.dot.model
        # We can't pre-compute the matrix because it depends on the bad pixel pattern.
        matrix = numpy.zeros((num, num))
        vector = numpy.zeros((num,))
        for ii in range(num):
            bg = bgModels[ii].getImage().getArray()[isGood]
            vector[ii] = numpy.dot(bg, array)
            for jj in range(ii, num):
                matrix[ii, jj] = numpy.dot(bg, bgModels[jj].getImage().getArray()[isGood])
                matrix[jj, ii] = matrix[ii, jj]
        return matrix, vector

    def solveComponents(self, matrices, vectors):
        """Solve for the background model component scales

        We solve the least-squares equation, summing the contributions
        from multiple elements (e.g., the different CCDs in an exposure).

        Parameters
        ----------
        matrices : `list` of `numpy.ndarray`
            Least-squares matrices ("Fisher matrices")
        vectors : `list` of `numpy.ndarray`
            Least-squares vectors ("right-hand sides")

        Returns
        -------
        solution : `numpy.ndarray`
            Background model component scales
        """
        matrix = sum(matrices, numpy.zeros_like(matrices[0]))
        vector = sum(vectors, numpy.zeros_like(vectors[0]))
        solution = afwMath.LeastSquares.fromNormalEquations(matrix, vector).getSolution()
        self.log.info("Sky component solution: %s" % (solution,))
        return solution

    def subtractComponents(self, image, bgModels, solution):
        """Subtract background model components

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Science image.
        bgModels : `list` of `lsst.afw.math.BackgroundList`
            Background model components.
        solution : array-like
            Scale factors for background model components.
        """
        assert len(bgModels) == len(solution)
        for scale, bg in zip(solution, bgModels):
            image.scaledMinus(scale, bg.getImage())



def interpolate1D(method, xSample, ySample, xInterp):
    """Interpolate in one dimension

    Interpolates the curve provided by `xSample` and `ySample` at
    the positions of `xInterp`. Automatically backs off the
    interpolation method to achieve successful interpolation.

    Parameters
    ----------
    method : `lsst.afw.math.Interpolate.Style`
        Interpolation method to use.
    xSample : `numpy.ndarray`
        Vector of ordinates.
    ySample : `numpy.ndarray`
        Vector of coordinates.
    xInterp : `numpy.ndarray`
        Vector of ordinates to which to interpolate.

    Returns
    -------
    yInterp : `numpy.ndarray`
        Vector of interpolated coordinates.

    """
    if len(xSample) == 0:
        return numpy.ones_like(xInterp)*numpy.nan
    try:
        return afwMath.makeInterpolate(xSample.astype(float), ySample.astype(float),
                                       method).interpolate(xInterp.astype(float))
    except:
        if method == afwMath.Interpolate.CONSTANT:
            # We've already tried the most basic interpolation and it failed
            return numpy.ones_like(xInterp)*numpy.nan
        newMethod = afwMath.lookupMaxInterpStyle(len(xSample))
        if newMethod == method:
            newMethod = afwMath.Interpolate.CONSTANT
        return interpolate1D(newMethod, xSample, ySample, xInterp)


class FocalPlaneBackgroundConfig(Config):
    """Configuration for FocalPlaneBackground

    Note that `xBins` and `yBins` are floating-point values, as
    the focal plane frame is usually defined in units of microns
    or millimetres rather than pixels. As such, their values will
    need to be revised according to each particular camera.
    """
    xSize = Field(dtype=float, default=2048, doc="Bin size in x")
    ySize = Field(dtype=float, default=2048, doc="Bin size in y")
    minFrac = Field(dtype=float, default=0.1, doc="Minimum fraction of bin size for good measurement")
    mask = ListField(dtype=str, doc="Mask planes to treat as bad",
                     default=["BAD", "SAT", "INTRP", "DETECTED", "DETECTED_NEGATIVE", "EDGE", "NO_DATA"])
    interpolation = ChoiceField(
        doc="how to interpolate the background values. This maps to an enum; see afw::math::Background",
        dtype=str, default="NATURAL_SPLINE", optional=True,
        allowed={
            "CONSTANT": "Use a single constant value",
            "LINEAR": "Use linear interpolation",
            "NATURAL_SPLINE": "cubic spline with zero second derivative at endpoints",
            "AKIMA_SPLINE": "higher-level nonlinear spline that is more robust to outliers",
            "NONE": "No background estimation is to be attempted",
        },
    )
    binning = Field(dtype=int, default=256, doc="Binning to use for CCD background model")


class FocalPlaneBackground(object):
    """Background model for a focal plane camera

    We model the background empirically with the "superpixel" method: we
    measure the background in each superpixel and interpolate between
    superpixels to yield the model.

    The principal difference between this and `lsst.afw.math.BackgroundMI`
    is that here the superpixels are defined in the frame of the focal
    plane of the camera.
    """
    @classmethod
    def fromCamera(cls, config, camera):
        """Construct from a camera object

        Parameters
        ----------
        config : `FocalPlaneBackgroundConfig`
            Configuration for measuring backgrounds.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera for which to measure backgrounds.
        """
        cameraBox = afwGeom.Box2D()
        for ccd in camera:
            for point in ccd.getCorners(afwCameraGeom.FOCAL_PLANE):
                cameraBox.include(point)

        width, height = cameraBox.getDimensions()
        # Offset so we run from zero
        offset = afwGeom.Extent2D(cameraBox.getMin())*-1
        # Add an extra pixel buffer on either side
        dims = afwGeom.Extent2I(int(numpy.ceil(width/config.xSize)) + 2,
                                int(numpy.ceil(height/config.ySize)) + 2)
        transform = (afwGeom.AffineTransform.makeTranslation(afwGeom.Extent2D(1, 1))*
                     afwGeom.AffineTransform.makeScaling(1.0/config.xSize, 1.0/config.ySize)*
                     afwGeom.AffineTransform.makeTranslation(offset))

        return cls(config, dims, transform)

    def __init__(self, config, dims, transform, values=None, numbers=None):
        """Constructor

        Parameters
        ----------
        config : `FocalPlaneBackgroundConfig`
            Configuration for measuring backgrounds.
        dims : `lsst.afw.geom.Extent2I`
            Dimensions for background samples.
        transform : `lsst.afw.geom.AffineTransform`
            Transformation from focal plane coordinates to sample coordinates.
        values : `lsst.afw.image.ImageF`
            Measured background values.
        numbers : `lsst.afw.image.ImageF`
            Number of pixels in each background measurement.
        """
        self.config = config
        self.dims = dims
        self.transform = transform

        if values is None:
            values = afwImage.ImageF(self.dims)
            values.set(0.0)
        else:
            values = values.clone()
        assert(values.getDimensions() == self.dims)
        self._values = values
        if numbers is None:
            numbers = afwImage.ImageF(self.dims)  # float for dynamic range and convenience
            numbers.set(0.0)
        else:
            numbers = numbers.clone()
        assert(numbers.getDimensions() == self.dims)
        self._numbers = numbers

    def __reduce__(self):
        return self.__class__, (self.config, self.dims, self.transform, self._values, self._numbers)

    def clone(self):
        return self.__class__(self.config, self.dims, self.transform, self._values, self._numbers)

    def addCcd(self, exposure):
        """Add CCD to model

        We measure the background on the CCD, and record the results
        in the model.  For simplicity, measurements are made in a box
        on the CCD corresponding to the warped coordinates of the
        superpixel rather than accounting for little rotations, etc.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            CCD exposure to measure
        """
        detector = exposure.getDetector()
        transform = detector.getTransformMap().get(detector.makeCameraSys(afwCameraGeom.FOCAL_PLANE))

        # Bin the exposure
        image = exposure.getMaskedImage()
        maskVal = image.getMask().getPlaneBitMask(self.config.mask)

        # Warp the binned image to the focal plane
        toSample = afwGeom.MultiXYTransform([transform, afwGeom.AffineXYTransform(self.transform)])

        warped = afwImage.ImageF(self._values.getBBox())
        warpedCounts = afwImage.ImageF(self._numbers.getBBox())
        width, height = warped.getDimensions()

        pixels = itertools.product(range(width), range(height))
        stats = afwMath.StatisticsControl()
        stats.setAndMask(maskVal)
        stats.setNanSafe(True)
        # Usually, iterating over individual pixels in python is bad, but there aren't many.
        for xx, yy in pixels:
            llc = toSample.reverseTransform(afwGeom.Point2D(xx - 0.5, yy - 0.5))
            urc = toSample.reverseTransform(afwGeom.Point2D(xx + 0.5, yy + 0.5))
            bbox = afwGeom.Box2I(afwGeom.Point2I(llc), afwGeom.Point2I(urc))
            bbox.clip(image.getBBox())
            if bbox.isEmpty():
                continue
            subImage = image.Factory(image, bbox)
            result = afwMath.makeStatistics(subImage, afwMath.MEANCLIP | afwMath.NPOINT, stats)
            mean = result.getValue(afwMath.MEANCLIP)
            num = result.getValue(afwMath.NPOINT)
            if not numpy.isfinite(mean) or not numpy.isfinite(num):
                continue
            warped.set(xx, yy, mean*num)
            warpedCounts.set(xx, yy, num)

        self._values += warped
        self._numbers += warpedCounts

    def toCcdBackground(self, detector, bbox):
        """Produce a background model for a CCD

        The superpixel background model is warped back to the
        CCD frame, for application to the individual CCD.

        Parameters
        ----------
        detector : `lsst.afw.cameraGeom.Detector`
            CCD for which to produce background model.
        bbox : `lsst.afw.geom.Box2I`
            Bounding box of CCD exposure.

        Returns
        -------
        bg : `lsst.afw.math.BackgroundList`
            Background model for CCD.
        """
        transform = detector.getTransformMap().get(detector.makeCameraSys(afwCameraGeom.FOCAL_PLANE))
        binTransform = afwGeom.AffineXYTransform(afwGeom.AffineTransform.makeScaling(self.config.binning))
        toSample = afwGeom.MultiXYTransform([binTransform, transform,
                                             afwGeom.AffineXYTransform(self.transform)])

        focalPlane = self.getStatsImage()
        fpNorm = afwImage.ImageF(focalPlane.getBBox())
        fpNorm.set(1.0)

        image = afwImage.ImageF(bbox.getDimensions()//self.config.binning)
        norm = afwImage.ImageF(image.getBBox())
        ctrl = afwMath.WarpingControl("bilinear")
        afwMath.warpImage(image, focalPlane, toSample.invert(), ctrl)
        afwMath.warpImage(norm, fpNorm, toSample.invert(), ctrl)
        image /= norm

        mask = afwImage.MaskU(image.getBBox())
        isBad = numpy.isnan(image.getArray())
        mask.getArray()[isBad] = mask.getPlaneBitMask("BAD")
        image.getArray()[isBad] = image.getArray()[~isBad].mean()

        return afwMath.BackgroundList(
            (afwMath.BackgroundMI(bbox, afwImage.makeMaskedImage(image, mask)),
             afwMath.stringToInterpStyle(self.config.interpolation),
             afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
             afwMath.ApproximateControl.UNKNOWN,
             0, 0, False)
            )

    def merge(self, other):
        """Merge with another FocalPlaneBackground

        This allows multiple background models to be constructed from
        different CCDs, and then merged to form a single consistent
        background model for the entire focal plane.

        Parameters
        ----------
        other : `FocalPlaneBackground`
            Another background model to merge.

        Returns
        -------
        self : `FocalPlaneBackground`
            The merged background model.
        """
        if (self.config.xSize, self.config.ySize) != (other.config.xSize, other.config.ySize):
            raise RuntimeError("Size mismatch: %s vs %s" % ((self.config.xSize, self.config.ySize),
                                                            (other.config.xSize, other.config.ySize)))
        if self.dims != other.dims:
            raise RuntimeError("Dimensions mismatch: %s vs %s" % (self.dims, other.dims))
        self._values += other._values
        self._numbers += other._numbers
        return self

    def __iadd__(self, other):
        """Merge with another FocalPlaneBackground

        Parameters
        ----------
        other : `FocalPlaneBackground`
            Another background model to merge.

        Returns
        -------
        self : `FocalPlaneBackground`
            The merged background model.
        """
        return self.merge(other)

    def getStatsImage(self):
        """Return the background model data

        This is the measurement of the background for each of the superpixels.
        """
        values = self._values.clone()
        values /= self._numbers
        thresh = self.config.minFrac*self.config.xSize*self.config.ySize

        isBad = self._numbers.getArray() < thresh
        isGood = ~isBad
        width, height = self.dims
        xIndices = numpy.arange(width, dtype=float)
        yIndices = numpy.arange(height, dtype=float)
        method = afwMath.stringToInterpStyle(self.config.interpolation)

        array = values.getArray()
        for y in range(height):
            if numpy.any(isBad[y, :]) and numpy.any(isGood[y, :]):
                array[y][isBad[y]] = interpolate1D(method, xIndices[isGood[y]], array[y][isGood[y]],
                                                   xIndices[isBad[y]])

        isBad = numpy.isnan(array)
        isGood = ~isBad
        for x in range(width):
            if numpy.any(isBad[:, x]) and numpy.any(isGood[:, x]):
                array[:, x][isBad[:, x]] = interpolate1D(method, yIndices[isGood[:, x]],
                                                         array[:, x][isGood[:, x]], yIndices[isBad[:, x]])

        return values
