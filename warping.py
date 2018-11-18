import numpy as np
from region.image import Image
from skimage.draw import polygon_perimeter, polygon
from scipy.interpolate import interp2d, dfitpack
import scipy.ndimage.morphology

def transf(H, res, normalize=True):
    if res.ndim == 1:
        vect = np.append(res, 1)
    else:
        vect = np.vstack((res.T, np.ones(len(res))))
    tfSourceCorners = np.dot(H, vect)

    if normalize:
        for i in range(tfSourceCorners.shape[1]):
            corner = tfSourceCorners[:, i]
            normCorner = np.dot(corner, 1 / (corner[2]))
            tfSourceCorners[:, i] = normCorner
    return tfSourceCorners[:-1, :].T


def getDimensions(imCornersCoords):
    # assuming imCornersCoords = [[width, height]]

    minWidth = int(np.floor(min(imCornersCoords[:, 0])))
    minHeight = int(np.floor(min(imCornersCoords[:, 1])))

    maxWidth = int(np.ceil(max(imCornersCoords[:, 0])))
    maxHeight = int(np.ceil(max(imCornersCoords[:, 1])))

    totalWidth = maxWidth - minWidth + 1
    totalHeight = maxHeight - minHeight + 1

    tup = totalHeight, totalWidth, minHeight, minWidth
    print(tup)

    return tup


def findAllCoords(tfSourceCorners):
    xCorners = tfSourceCorners[:, 0]
    yCorners = tfSourceCorners[:, 1]

    flatMinX = int(np.floor(min(xCorners)))
    flatMinY = int(np.floor(min(yCorners)))
    newXCorners = xCorners - flatMinX
    newYCorners = yCorners - flatMinY

    innerCoordsR, innerCoordsC = polygon(newXCorners, newYCorners)
    finInnerCoordsR, finInnerCoordsC = innerCoordsR + flatMinX, innerCoordsC + flatMinY

    perimCoordsR, perimCoordsC = polygon_perimeter(xCorners, yCorners)
    allCoordsR = np.append(finInnerCoordsR, perimCoordsR)
    allCoordsC = np.append(finInnerCoordsC, perimCoordsC)

    #     allCoordsR = finInnerCoordsR
    #     allCoordsC = finInnerCoordsC

    #     allCoordsR = perimCoordsR
    #     allCoordsC = perimCoordsC

    targetPs = np.vstack((allCoordsR, allCoordsC)).T

    return targetPs


class WarpedImageInfo:

    def __init__(self, corners, targetPs, rvals, gvals, bvals, iavals):
        self.corners = corners
        self.targetPs = targetPs
        self.rvals = rvals
        self.gvals = gvals
        self.bvals = bvals
        self.iavals = iavals

    def createImageObj(self, newImage=None, minH=0, minW=0, name=""):
        if newImage == None:
            finH, finW, minH, minW = getDimensions(self.corners)
            newImage = Image(name, np.zeros((finH, finW, 4)))
        newImage.imArr[self.targetPs[:, 1] - minH, self.targetPs[:, 0] - minW, 0] = self.rvals
        newImage.imArr[self.targetPs[:, 1] - minH, self.targetPs[:, 0] - minW, 1] = self.gvals
        newImage.imArr[self.targetPs[:, 1] - minH, self.targetPs[:, 0] - minW, 2] = self.bvals
        newImage.imArr[self.targetPs[:, 1] - minH, self.targetPs[:, 0] - minW, 3] = self.iavals
        newImage.polyCorners = self.corners
        return newImage


def warpTo(imSource, sourcePts, targetPts, padding=0):
    H = computeH(sourcePts, targetPts)
    sourceCorners = Image.getCorners(imSource, padding)
    tfSourceCorners = transf(H, sourceCorners)

    print("sourceCorners:", sourceCorners)
    print("tfSourceCorners:", tfSourceCorners)

    targetPs = findAllCoords(tfSourceCorners)

    invH = np.linalg.inv(H)

    sourcePs = transf(invH, targetPs)

    source_Rchan = interp2d(range(imSource.width), range(imSource.height), imSource.imArr[:, :, 0])
    source_Gchan = interp2d(range(imSource.width), range(imSource.height), imSource.imArr[:, :, 1])
    source_Bchan = interp2d(range(imSource.width), range(imSource.height), imSource.imArr[:, :, 2])

    _alpHelp = np.ones(imSource.imArr.shape[:2])
    _alpHelp[_alpHelp.shape[0] // 2, _alpHelp.shape[1] // 2] = 0

    _alp2 = scipy.ndimage.morphology.distance_transform_edt(_alpHelp)
    weightedAlph = 1 - (_alp2 / np.amax(_alp2))
    #     print(np.amin(weightedAlph), np.amax(weightedAlph))
    #     printImage("test.png", weightedAlph)

    source_iAchan = interp2d(range(imSource.width), range(imSource.height), weightedAlph)

    autoInterp = lambda f, xs, ys: dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], xs, ys)[0]

    Rvals = autoInterp(source_Rchan, sourcePs[:, 0], sourcePs[:, 1])
    Gvals = autoInterp(source_Gchan, sourcePs[:, 0], sourcePs[:, 1])
    Bvals = autoInterp(source_Bchan, sourcePs[:, 0], sourcePs[:, 1])
    iAvals = autoInterp(source_iAchan, sourcePs[:, 0], sourcePs[:, 1])

    return WarpedImageInfo(tfSourceCorners, targetPs, Rvals, Gvals, Bvals, iAvals)


def warpProcess(imBase, imBasePts, imgList, padding=0):
    warpedResults = []
    for im in imgList:
        imPts = im.pts

        warpedResults.append(warpTo(im, imPts, imBasePts, padding))

    allCorners = np.vstack([warped.corners for warped in warpedResults])
    print(allCorners)
    finH, finW, minH, minW = getDimensions(allCorners)
    # finalImage = Image("finalImage", np.zeros((finH, finW, 3)))

    newWarpedImages = []
    # for warped in warpedResults:
    for i in range(len(warpedResults)):
        warped = warpedResults[i]
        newImage = Image("finalImage", np.zeros((finH, finW, 4)))
        newWarpedImages.append(warped.createImageObj(newImage, minH=minH, minW=minW))

    return newWarpedImages


def computeH(startPts, endPts):
    assert len(startPts) == len(endPts)
    A, b = createAb(startPts, endPts)

    x = np.linalg.lstsq(A, b)
    result = np.append(x[0], 1)
    return result.reshape(3, 3)


def createAb(startPts, endPts):
    result = np.zeros((len(startPts) * 2, 8))
    b = np.zeros(len(result))
    for i in range(len(startPts)):
        startPoint = startPts[i]
        endPoint = endPts[i]

        # first row set
        result[i * 2, 0] = -1 * startPoint[0]
        result[i * 2, 1] = -1 * startPoint[1]
        result[i * 2, 2] = -1

        result[i * 2, 6] = startPoint[0] * endPoint[0]
        result[i * 2, 7] = startPoint[1] * endPoint[0]
        b[i * 2] = -endPoint[0]

        # second row set
        result[i * 2 + 1, 3] = -1 * startPoint[0]
        result[i * 2 + 1, 4] = -1 * startPoint[1]
        result[i * 2 + 1, 5] = -1

        result[i * 2 + 1, 6] = startPoint[0] * endPoint[1]
        result[i * 2 + 1, 7] = startPoint[1] * endPoint[1]
        b[i * 2 + 1] = -endPoint[1]
    # result[-1, -1] = 1

    return result, b