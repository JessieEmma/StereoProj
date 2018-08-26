import glob
import os
import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
images = glob.glob('../left/*.jpg')

def find_points():
    """
    find the 3D points and corresponding 2D points of corners in the chess board
    :return: 3D object points in the world and 2D image points in the image plane
    """

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # row by row, left to right in every row, z value defines 0

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6))

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners2 = np.squeeze(corners2, axis=1) # Remove single-dimensional of axix 1
        imgpoints.append(corners2)  # add the corner points corresponding to the 3D points

        objpoints.append(objp)

    imageSize = gray.shape[::-1]
    return objpoints, imgpoints, imageSize

def calibration():
    """
    using found points to calibrate
    :return:
    """
    objpoints, imgpoints, imageSize = find_points()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageSize, None, None)
    return ret, mtx, dist, rvecs, tvecs


def undistortion(mtx, dist, imgname, outputname=None):
    """
    using calibration parameters to undistort an image
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param imgname: the image file to be undistorted
    :param outputname: the output file to be show undistortion result
    :return:
    """

    img = cv2.imread(imgname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop and save the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    if (outputname == None):
        if (not os.path.exists("../result")):
            os.makedirs("../result")
        outputname = "../result/"+ imgname.split("/")[1].split(".")[0]+"_undistortion.png"
    cv2.imwrite(outputname, dst)

if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibration()
    print("camera matrix is ")
    print(mtx)
    print("distortion coeffients is ")
    print(dist)
    print("rotation matrix is ")
    print(rvecs)
    print("translation matrix is ")
    print(tvecs)
    undistortion(mtx, dist, "../left/left01.jpg")