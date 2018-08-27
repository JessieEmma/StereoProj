# Stereo Project
This repository contains the code for "Stereo Project".

## Environment Setup
### Language 
* [Python 3.6](https://www.python.org/downloads/)
### Tool
* [pip 18.0](https://pypi.org/project/pip)
### Packages
* [OpenCV 3.4.2](https://opencv.org/releases.html)
*  numpy 1.15.0

### Vitual Environment setup
1. Set up a new virtual environmrnt 
```
$ virtual venv   <tab><tab> # venv :the name of virtual environment directory name
```
2. Choose an interpreter for the environment
```
$ virtualenv -p /usr/bin/python2.7 venv　　<tab><tab>　　# -p parameter determines the path of Python interpreter
```
3. Start use this environment
```
$ source venv/bin/activate　 <tab><tab> # source command
```
4. Install the required packages for this project
```
$ pip install -r requirement.txt     <tab><tab>  # -r parameter is the path of requirements.txt of this project
```
## Camera Basics 
[calib.py](https://github.com/JessieEmma/StereoProj/blob/master/CameraBasics/calib.py) contains calibration of the camera with OpenCV functions for Q6 and Q7.
 ```
 find_points  helps establish the 3D points and find corresponding 2D points of corners in the chess board
 calibration  is integrated with find_points and cv2.calibrateCamera
 undistortion  is integrated with cv2.undistort
 undistortPoint  the transformation from the distorted point to the undistorted point for Q4
 ```
[calib_imp.py](https://github.com/JessieEmma/StereoProj/blob/master/CameraBasics/calib_imp.py) contains my implementation of [Zhang's method](https://www.researchgate.net/publication/3193178_A_Flexible_New_Technique_for_Camera_Calibration) for Q8.
 ```
 estimate_homography  helps find the homography for image points and camera points
 getMatrixB   helps get B=A-T·A-1
 extract_intrinsic  helps get A from B=A-T·A-1
 compute_extrinsic  helps get [R|t] from A and H
 get_radialCoefficients  helps compute distortion coeffients 
 img_coordinates  find image points 
 calibrationImplement   is integrated with all parameters getting function
 ```
## Binocular Basics
[bino_calib.py](https://github.com/JessieEmma/StereoProj/blob/master/BinocularBasics/bino_calib.py) contains calibration of the binocular cameras for Q12.
 ```
 find_points   helps get the 2D points of one of the plane, which is called for each direction respectively
 calib_two   is integrated with find_points to calibrate
 ```
[rectification.py](https://github.com/JessieEmma/StereoProj/blob/master/BinocularBasics/rectification.py) contains rectification of the left and the right images with the calibration results for Q14.
 ```
 undistortion   is integrated with cv2.initUndistortRectifyMap and cv2.remap, to some extent, which equals cv2.undistort 
 recti   is integrated with calibration to rectify these points and show
 ```

## Stereo Matching
[disparity.py](https://github.com/JessieEmma/StereoProj/blob/master/StereoMatching/disparity.py) contains disparity computation by OpenCV functions based on SGBM for Q17.
 ```
 get_img   helps open two corresponding images
 createSGBM   helps set up a stereo model
 get_disparity   helps get the disparity using the images and model
 ```


## Images
[left](https://github.com/JessieEmma/StereoProj/tree/master/left) directory contains the images taken by left camera

[right](https://github.com/JessieEmma/StereoProj/tree/master/right) directory contains the images taken by right camera

[result](https://github.com/JessieEmma/StereoProj/tree/master/result) directory will save the result images 
 