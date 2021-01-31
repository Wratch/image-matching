import cv2
import numpy as np
import argparse


def parser():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--cropped", type=str, help="Path to cropped image of the star map",
                        default='Assets/Small_area_rotated.png')
    parser.add_argument("--original", type=str, help="Path to the original image of the star map",
                        default='Assets/StarMap.png')
    return parser.parse_args()


def feature_matching():
    img1 = cv2.imread(cropped_image, 0)  # Original Image
    img2 = cv2.imread(original_image, 0)  # Random cropped Image

    orb = cv2.ORB_create(nfeatures=100000, WTA_K=2, edgeThreshold=0, patchSize=25)

    # Detects key points and computes the descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)

    # Orb matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = matches[:50]  # Threshold

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    height, width = img1.shape[:2]
    pts = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(pts, M)
    dst += (width, 0)  # adding offset

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

    img_result = cv2.polylines(img3, [np.int32(dst)], True, (255, 255, 255), 3, cv2.LINE_AA)

    coord_result = dst.tolist()
    print(coord_result)

    cv2.imshow("result", img_result)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parser()
    cropped_image = args.cropped
    original_image = args.original
    feature_matching()
