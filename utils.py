from skimage.io import imsave, imshow, show
import numpy as np
import datetime
import os
import skimage.io as skio
import skimage as sk
import imageio
from skimage.transform import resize

import matplotlib.pyplot as plt

fOutputDirectory = "output_imgs"
fFormat = ".jpg"

#IMAGE IO
TESTING_DIR = "trash_imgs"

def timeStampPath(path):
    currTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    finalPath = "%s_%s" % (currTime, path)

    return finalPath

def printImage(name, im, disp=False, directory = fOutputDirectory):
    # save the image

    finalPath = directory + "/"+ timeStampPath(name)
    imsave(finalPath, im)

    print("File saved to:", finalPath)

    # display the image
    if disp:
        viewImage(im)

def testImage(path, im, disp=True):
    finalPath = path

    printImage(TESTING_DIR, finalPath, im, disp)

def viewImage(im):
    imshow(im)
    show()

def grayscale2RGB(im):
    assert im.ndim == 2

    return np.dstack((np.dstack((im, im)), im))

def readImageNName(impath1):
    imname1 = "".join(os.path.basename(impath1).split(".")[:-1])
    im1 = skio.imread(impath1)
    im1 = sk.img_as_float(im1)
    return im1, imname1

def convertToGif(fileName, lstIm, resizeFactor = None):
    return convertToGifArr(fileName, [im.imArr for im in lstIm], resizeFactor)

def convertToGifArr(fileName, lstImArr, resizeFactor = None):
    if resizeFactor == None:
        resizeFactor = 1
    else:
        assert type(resizeFactor) == float
        for i in range(len(lstImArr)):
            imArr = lstImArr[i]
            lstImArr[i] = resize(imArr, (imArr.shape[0] * resizeFactor, imArr.shape[1] * resizeFactor))
    imageio.mimsave('./%s/%s-r%s.gif' % (fOutputDirectory, timeStampPath(fileName), str(resizeFactor)), lstImArr)
    return

def backForth(result):
    return result + result[::-1]