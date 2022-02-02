import cv2
import numpy as np


def findHomography(img1, img2):

    sift = cv2.SIFT_create()
    keypoint1, descriptor1 = sift.detectAndCompute(img1, None)
    keypoint2, descriptor2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptor1, descriptor2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)

    source_points = np.float32([keypoint1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    destination_points = np.float32([keypoint2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)

    return M, source_points, destination_points, mask


def laplacianPyramid(img1, img2, mask, levels):

    gauss_1 = img1.copy()
    gauss_2 = img2.copy()
    gauss_mask = mask.copy()

    gauss_pyramid_1 = [gauss_1]
    gauss_pyramid_2 = [gauss_2]
    gauss_pyramid_mask = [gauss_mask]

    for i in range(levels):
        gauss_1 = cv2.pyrDown(gauss_1)
        gauss_2 = cv2.pyrDown(gauss_2)
        gauss_mask = cv2.pyrDown(gauss_mask)
        gauss_pyramid_1.append(np.float32(gauss_1))
        gauss_pyramid_2.append(np.float32(gauss_2))
        gauss_pyramid_mask.append(np.float32(gauss_mask))

    laplacian_pyramid_1 = [gauss_pyramid_1[levels - 1]]
    laplacian_pyramid_2 = [gauss_pyramid_2[levels - 1]]
    gauss_pyramid_mask_2 = [gauss_pyramid_mask[levels - 1]]
    for i in range(levels - 1, 0, -1):
        L1 = np.subtract(gauss_pyramid_1[i - 1], cv2.pyrUp(gauss_pyramid_1[i]))
        L2 = np.subtract(gauss_pyramid_2[i - 1], cv2.pyrUp(gauss_pyramid_2[i]))
        laplacian_pyramid_1.append(L1)
        laplacian_pyramid_2.append(L2)
        gauss_pyramid_mask_2.append(gauss_pyramid_mask[i - 1])

    LS = []
    for l1, l2, gm in zip(laplacian_pyramid_1, laplacian_pyramid_2, gauss_pyramid_mask_2):
        ls = l1 * gm + l2 * (1 - gm)
        LS.append(ls)

    ls_ = LS[0]
    for i in range(1, levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_


def mergingImages(img1, img2, img3):

    img1 = cv2.copyMakeBorder(img1, 200, 200, 500, 500, cv2.BORDER_CONSTANT)
    (M1, points1, points2, mask1) = findHomography(img3, img1)
    (M2, points3, points4, mask2) = findHomography(img2, img1)

    m1 = np.ones_like(img3, dtype='float32')
    m2 = np.ones_like(img2, dtype='float32')

    output1 = cv2.warpPerspective(img3, M1, (img1.shape[1], img1.shape[0]))
    output2 = cv2.warpPerspective(img2, M2, (img1.shape[1], img1.shape[0]))
    output3 = cv2.warpPerspective(m1, M1, (img1.shape[1], img1.shape[0]))
    output4 = cv2.warpPerspective(m2, M2, (img1.shape[1], img1.shape[0]))

    laplacian_pyramid_1 = laplacianPyramid(output1, img1, output3, 1)

    laplacian_pyramid_2 = laplacianPyramid(output2, laplacian_pyramid_1, output4, 1)

    return laplacian_pyramid_2


if __name__ == '__main__':

    image1 = cv2.imread("building1.jpg")
    image2 = cv2.imread("building2.jpg")
    image3 = cv2.imread("building3.jpg")

    result = mergingImages(image2, image1, image3)

    cv2.imwrite("best_result.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
