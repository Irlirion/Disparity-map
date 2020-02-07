from time import time

import cv2
import numpy as np
import glob


def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    # print("hello")
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


left_imgs = glob.glob('data/I1*.png')
right_imgs = glob.glob('data/I2*.png')

count = 0

out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (1344, 372))

while count < 111:
    start_time = time()

    imgL = adjust_gamma(cv2.imread(left_imgs[count]), 1)
    imgR = adjust_gamma(cv2.imread(right_imgs[count]), 1)

    window_size = 5
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    print('computing disparity...')
    displ = left_matcher.compute(imgL, imgR).astype(np.float32) / 16
    dispr = right_matcher.compute(imgR, imgL).astype(np.float32) / 16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    # ret, thresh1 = cv2.threshold(filteredImg, 100, 255, cv2.THRESH_BINARY)

    new_filter = np.zeros((372, 1344, 3), dtype='uint8')
    for i, c in enumerate(filteredImg):
        for j, k in enumerate(c):
            new_filter[i][j][0] = k
            new_filter[i][j][1] = k
            new_filter[i][j][2] = k

    print(f"{count}: Done in {round(time() - start_time, 2)}s")

    out.write(new_filter)
    # Display the resulting frame
    cv2.imshow('Disparity Map', new_filter)
    # cv2.imshow('Frame1',thresh1)
    # cv2.imshow('Frame2',imgR)
    # Press Q on keyboard to  exit

    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
