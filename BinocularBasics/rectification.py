import glob
import cv2
from BinocularBasics import bino_calib

def undistortion(src, mtx, dist, R, P, image_size):
    """
    undistort the image by using rectified parameters
    :param src:
    :param mtx:
    :param dist:
    :param R:
    :param P:
    :param image_size:
    :return:
    """
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, R, P, image_size, cv2.CV_16SC2)

    dst = cv2.remap(src, map1, map2, cv2.INTER_LINEAR)

    return dst

def recti(is_show=True):
    """
    Rectify the two-direction images
    :param is_show: choose to show images or not
    :return: the baseline of horizontal strereo
    so that each pixel's epipolar line is horizontal and at the same vertical position as that pixel.
    """
    mtx1, dist1, mtx2, dist2, R, t, E, F, image_size = bino_calib.calib_two()
    print("matrix of camera 1")
    print(mtx1)
    print("matrix of camera 2")
    print(mtx2)
    R1, R2, P1, P2, Q, valiROI1, valiROI2 = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, image_size, R, t, alpha=1)
    print("R1 is")
    print(R1)
    print("R2 is")
    print(R2)
    print("P1 is")
    print(P1)
    print("P2 is")
    print(P2)
    left_images = glob.glob(bino_calib.left_img_path)
    right_images = glob.glob(bino_calib.right_img_path)

    choose_image = 1
    left_image = left_images[0]
    right_image = right_images[0]
    img_src = cv2.imread(left_image)
    img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # undistort the image
    img_dst = undistortion(img_src_gray, mtx1, dist1, R1, P1, image_size)

    if is_show:
        cv2.imshow(left_image, img_src_gray)
        cv2.imshow("rectified_" + left_image, img_dst)
        cv2.waitKey(0)

    img_src = cv2.imread(right_image)
    img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # undistort the image
    img_dst = undistortion(img_src_gray, mtx2, dist2, R2, P2, image_size)

    if is_show:
        cv2.imshow(right_image, img_src_gray)
        cv2.imshow("rectified_" + left_image, img_dst)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    b = (P1[0, 3] - P2[0, 3])/P1[0, 0]
    return b



if __name__ == "__main__":
    # b = recti()
    b = recti(is_show=False)
    print("The b of baseline (b, 0, 0) is")
    print(b)