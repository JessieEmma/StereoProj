import cv2
import bino_calib
import glob

def undistortion(img_src, mtx, dist, R, new_mtx, image_size):
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, R, new_mtx, image_size, cv2.CV_32FC1)

    dst = cv2.remap(img_src, map1, map2, cv2.INTER_LINEAR)
    return dst

def recti(is_show=True):
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

    for (left_image, right_image) in zip(left_images, right_images):
        img_src = cv2.imread(left_image)
        img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

        # img_dst = cv2.undistort(img_src_gray, mtx1, dist1) a combination of initUndistortRectifyMap (with unity R ) and remap (with bilinear interpolation).
        img_dst = undistortion(img_src_gray, mtx1, dist1, R, P1, image_size)

        if is_show:
            cv2.imshow(left_image, img_src_gray)
            cv2.imshow("rectified_"+left_image, img_dst)
            cv2.waitKey(0)

        img_src = cv2.imread(right_image)
        img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

        img_dst = undistortion(img_src_gray, mtx1, dist1, R, P1, image_size)

        if is_show:
            cv2.imshow(right_image, img_src_gray)
            cv2.imshow("rectified_"+left_image, img_dst)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    recti()