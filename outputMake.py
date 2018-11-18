import numpy as np
import skimage.filters

def alphaBlendingOutput(imList2):
    assert len(imList2) > 0

    imList = [x.imArr for x in imList2]

    im1 = imList[0]
    totalAlphas = np.zeros(im1.shape[:2])
    for im in imList:
        totalAlphas += im[:, :, 3]
    totalAlphas = totalAlphas
    #     printImage("ta.png", totalAlphas / np.amax(totalAlphas))
    totalAlphas[totalAlphas == 0] = 1

    finalImage = np.zeros((im1.shape[0], im1.shape[1], 3))
    # div3 = np.dstack((totalAlphas, np.dstack((totalAlphas, totalAlphas))))
    # print(div3.shape)
    # print(finalImage.shape)
    for im in imList:
        weights = np.divide(im[:, :, 3], totalAlphas)
        # finalImage[:,:,1] += np.multiply(im[:,:,2], weights)
        weights3 = np.dstack((weights, np.dstack((weights, weights))))
        finalImage += np.multiply(im[:, :, :3], weights3)
    finalImage = np.clip(finalImage, 0, 1)
    return finalImage


### laplacian

def multiResBlendOp(im1, im2, mask, levels, sigma):
    assert im1.shape == im2.shape == mask.shape

    L1 = laplacianPyrOp_3D(im1, levels, sigma)
    L2 = laplacianPyrOp_3D(im2, levels, sigma)
    LM = gaussStackOp_3D(mask, levels, sigma)

    #     for i in range(len(LM)):
    #         printImage(str(i) + ".png", LM[i])

    #     for i in range(len(L1)):
    #         printImage(str(i) + ".png", L1[i])
    LM1 = LM
    LM2 = (1 - LM1)

    L1_post = LM1 * L1
    L2_post = LM2 * L2

    finalL = L1_post + L2_post

    tes = np.zeros(L1[0].shape)

    for i in range(len(L1)):
        tes += finalL[i]

    tes2 = np.clip(tes, -1, 1)

    return tes2

def laplacianPyrOp_3D(im, levels, sigma, scaleB = False):
    gaussStack = gaussStackOp_3D(im, levels, sigma)

    for i in range(levels):
        res = gaussStack[i] - gaussStack[i+1]
        if scaleB:
            finalCurrLayer = (res - res.min()) / (res.max() - res.min())
        else:
            finalCurrLayer = res
        gaussStack[i] = finalCurrLayer

    return gaussStack

def gaussStackOp_3D(im, levels, sigma):
    assert levels > 0
    #inclusive of original img, at layer indexed 0

    result = []
    # newLayer = (lambda: np.zeros(im.shape))

    for i in range(levels+1):
        if i == 0:
            result.append(im)
            continue
        # currLayer = newLayer()
        currLayer = skimage.filters.gaussian(result[i - 1], sigma, mode="constant")
        result.append(currLayer)

    return np.array(result)

def laplacianBlending(im1, im2, level, sig):
    imMask = np.zeros(im1.shape)
    # im1 == 0

    imMask[np.any(im1, 2), 0] = 1
    imMask[np.any(im1, 2), 1] = 1
    imMask[np.any(im1, 2), 2] = 1

    #### cheatsy doodle, the woooooorst way of doing this
    invImMask = 1 - imMask
    invImMask2 = skimage.filters.gaussian(invImMask, sig)
    for i in range(3):
        invImMask2 += skimage.filters.gaussian(invImMask, sig)
    imMask2 = 1 - invImMask2
    imMask2 = np.clip(imMask2, 0, 1)

    res = multiResBlendOp(im1, im2, imMask2, level, sig)

    return res