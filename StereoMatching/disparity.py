import cv2
import numpy as np

def get_img(left_img_path, right_img_path):
    """
    Get the image of the left and right camera
    :param left_img_path:
    :param right_img_path:
    :return:
    """
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    return left_img, right_img

def createSGBM(window_size=3, min_disp=16, max_disp=112):
    """
    :param window_size:
    :param min_disp: the minimun value of disparity
    :param max_disp: the maximun values of disparity
    :return:
    """

    num_disp = max_disp - min_disp

    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=16,
                                  P1=8 * 3 * window_size ** 2,
                                  P2=32 * 3 * window_size ** 2,
                                  disp12MaxDiff=1,
                                  uniquenessRatio=10,
                                  speckleWindowSize=100,
                                  speckleRange=32
                                  )

    return stereo


def get_disparity(left_img, right_img, stereo, min_disp=16, max_disp=112, is_show=True):
    """
    Get disparity of the two images
    :param left_img:
    :param right_img:
    :param stereo:
    :param min_disp:
    :param max_disp:
    :param is_show:
    :return:
    """
    disp = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    num_disp = max_disp - min_disp
    if is_show:
        cv2.imshow('left', left_img)
        cv2.imshow('right', right_img)
        cv2.imshow('disparity', (disp - min_disp)/ num_disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return disp


if __name__ == "__main__":
    stereo = createSGBM()
    left_img, right_img = get_img("../left/left01.jpg", "../right/right01.jpg")
    disp = get_disparity(left_img, right_img, stereo)
    print("disparity")
    print(disp)