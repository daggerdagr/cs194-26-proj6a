from utils import *

class Image:

    def __init__(self, name, imArr):
        self.name = name
        self.imArr = imArr
        self.height, self.width = imArr.shape[:2]

    def fromPath(pathName, resizeFactor=1.0):
        imArr, imName = readImageNName(pathName)
        if imArr.ndim == 2:
            h, w = imArr.shape
            newIm = np.empty((h, w, 3))
            newIm[:, :, 0] = imArr
            newIm[:, :, 1] = imArr
            newIm[:, :, 2] = imArr
            imArr = newIm
        if resize != 1.0:
            imArr = resize(imArr, (imArr.shape[0] * resizeFactor, imArr.shape[1] * resizeFactor))
        result = Image(imName, imArr)
        return result

    def getCorners(image, padding = 0):
        imArr = image.imArr
        return np.array(
            [(0 - padding, 0 - padding), (0 - padding, imArr.shape[0] + padding - 1), (imArr.shape[1] + padding - 1, imArr.shape[0] + padding - 1), (imArr.shape[1] + padding - 1, 0 - padding)])

    def corners(self):
        return Image.getCorners(self.imArr)

    def readCoordsIn(self, pathName):
        self.pts = Image.readCoords(pathName)

    def readCoords(pathName):
        fileString = fileToString(pathName)
        coords = np.array([[float(n) for n in entry.split(",")] for entry in fileString.split("\n")])
        return coords
