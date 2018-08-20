import cv2
import numpy as np
import glob

left_img_path = 'left/*.jpg'
right_img_path = 'right/*.jpg'

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

def find_points(img_path):
    """
    Get the 2D points of one of the plane
    :param img_path: all images path
    :return: 2D points in pictures
    """

    imgpoints = []  # 2d points in one of the image planes.
    images = glob.glob(img_path)

    image_num = len(images)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6))

        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        corners2 = np.squeeze(corners, axis=1)  # Remove single-dimensional of axix 1
        imgpoints.append(corners2)  # add the corner points corresponding to the 3D points

    imageSize = gray.shape[::-1]

    return imgpoints, imageSize, image_num

def calib_two():
    """
    Use two-camera recorded photos to get the position relationo of two camera
    :return: camera matrix and distrotion coefficients of the two, and the rotation matrix and translation from the left to the right,
    the essential matrix and foundamental matrix
    """
    objpoints = []  # 3d point in real world space

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # row by row, left to right in every row, z value defines 0

    # get 2D image plane points of the left and right
    left_imgpoints, left_image_size, left_imgnum = find_points(left_img_path)
    right_imgpoints, right_image_size, right_imgnum = find_points(right_img_path)


    for i in range(left_imgnum):
        objpoints.append(objp)

    flags = (cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST)

    ret, mtx1, dist1, mtx2, dist2, R, t, E, F = cv2.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints, None, None, None, None, left_image_size, flags=flags)

    return mtx1, dist1, mtx2, dist2, R, t, E, F, left_image_size

if __name__ == '__main__':
    mtx1, dist1, mtx2, dist2, R, t, E, F, image_size = calib_two()
    print("camera matrix of left is ")
    print(mtx1)
    print("distortion coeffients of left is ")
    print(dist1)
    print("camera matrix of right is ")
    print(mtx2)
    print("distortion coeffients of right is ")
    print(dist2)
    print("rotation matrix is ")
    print(R)
    print("translation matrix is ")
    print(t)
    print("Essential matrix is ")
    print(E)
    print("Foundamental matrix is ")
    print(F)