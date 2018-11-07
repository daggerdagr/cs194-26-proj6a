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