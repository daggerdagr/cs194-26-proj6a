from region.image import Image
from warping import *
from utils import *
from outputMake import *

def rectifyLowerSproul():
    print("Rectify Lower Sproul, Start")
    imRect = Image.fromPath("./sample_imgs/lower_sproul_00.jpg", 0.25)
    imRect.readCoordsIn("./sample_imgs/rect_orig_lower_sproul_00.txt")

    targPts = Image.readCoords("./sample_imgs/rect_targ_lower_sproul_00.txt")

    res = warpTo(imRect, imRect.pts, targPts)
    im = res.createImageObj()

    im.imArr[:,:,3] = 1
    printImage("rectified_lowerSproul.png", im.imArr)
    print("Rectify Lower Sproul, Done")

def mosaicLowerSproul():
    print("Mosaic Lower Sproul, Start")

    imSproul0 = Image.fromPath("./sample_imgs/lower_sproul_00.jpg", 0.25)
    imSproul0.readCoordsIn("./sample_imgs/2_lower_sproul_00.txt")
    imSproul0.pts = imSproul0.pts * 0.25
    imSproul1 = Image.fromPath("./sample_imgs/lower_sproul_01.jpg", 0.25)
    imSproul1.readCoordsIn("./sample_imgs/2_lower_sproul_01.txt")
    imSproul1.pts = imSproul1.pts * 0.25

    imgList = [imSproul0, imSproul1]
    imBase = imSproul1
    imBasePts = imSproul1.pts

    result = warpProcess(imBase, imBasePts, imgList, 0)

    printImage("alphaBlend_lowerSproul.png", alphaBlendingOutput(result))

    im0 = result[0].imArr[:,:,:3]
    im1 = result[1].imArr[:,:,:3]
    printImage("lapBlend_lowerSproul.png", laplacianBlending(im0, im1, 5, 20))
    print("Mosaic Lower Sproul, Done")


def mosaicRoofEsh():
    print("Mosaic Roof Esh, Start")
    imRoof0 = Image.fromPath("./sample_imgs/roof_esh_00.jpg", 0.25)
    imRoof0.readCoordsIn("./sample_imgs/1_roof_esh_00.txt")
    imRoof0.pts = imRoof0.pts * 0.25
    imRoof1 = Image.fromPath("./sample_imgs/roof_esh_01.jpg", 0.25)
    imRoof1.readCoordsIn("./sample_imgs/1_roof_esh_01.txt")
    imRoof1.pts = imRoof1.pts * 0.25

    imgList = [imRoof0, imRoof1]
    imBase = imRoof1
    imBasePts = imRoof1.pts

    result2 = warpProcess(imBase, imBasePts, imgList, 0)

    printImage("alphaBlend_roofEsh.png", alphaBlendingOutput(result2))

    im0 = result2[0].imArr[:, :, :3]
    im1 = result2[1].imArr[:, :, :3]
    printImage("lapBlend_roofEsh.png", laplacianBlending(im0, im1, 5, 20))

    print("Mosaic Roof Esh, Done")

def mosaicFirstEsh():
    print("Mosaic Roof Esh, Start")
    imRoof0 = Image.fromPath("./sample_imgs/first_esh_00.jpg", 0.25)
    imRoof0.readCoordsIn("./sample_imgs/first_esh_00.txt")
    imRoof0.pts = imRoof0.pts * 0.25
    imRoof1 = Image.fromPath("./sample_imgs/first_esh_01.jpg", 0.25)
    imRoof1.readCoordsIn("./sample_imgs/first_esh_01.txt")
    imRoof1.pts = imRoof1.pts * 0.25

    imgList = [imRoof0, imRoof1]
    imBase = imRoof1
    imBasePts = imRoof1.pts

    result2 = warpProcess(imBase, imBasePts, imgList, 0)

    printImage("alphaBlend_firstEsh.png", alphaBlendingOutput(result2))
    im0 = result2[0].imArr[:,:,:3]
    im1 = result2[1].imArr[:,:,:3]
    printImage("lapBlend_firstEsh.png", laplacianBlending(im0, im1, 5, 20))

    print("Mosaic Roof Esh, Done")