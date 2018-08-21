import numpy as np

from CameraBasics import calib


def estimate_homography(objpoints, imgpoints):
    """
    Implementation of Appendix A---Estimation of the Homography
    :param objpoints: 3D model points in one picture
    :param imgpoints: image points in one picture
    :return:
    """

    # since always Z = 0, ordinary coorinate transforming to homogeneous, we ignore Z and add one dimension of 1
    objpoints[:, 2] = 1

    n = objpoints.shape[0]
    L = np.zeros((2*n, 9))

    # initial guess
    for i in range(n):
        u = imgpoints[i, 0]
        v = imgpoints[i, 1]
        L[i*2, :3] = objpoints[i]
        L[i*2, 6:] = -(u * objpoints[i])
        L[i*2+1, 3:6] = objpoints[i]
        L[i*2+1, 6:] = -(v * objpoints[i])
    # equivalently computing
    eigenvalue, eigenvec = np.linalg.eig(np.dot(L.T, L))

    # find the least eigenvector
    least_eigen_idx = np.argmin(eigenvalue)
    least_eigen_vec = eigenvec[:, least_eigen_idx]
    H = np.reshape(least_eigen_vec, (3,3))

    return H


def getMatrixB(Hs):
    """
    Implementation of 3.1 Closed-form solution
    :param Hs: homography lists of object points in n pictures
    :return: matrix B, which is the input for extracting intrinsic
    """
    n = len(Hs)

    def getv(h, i, j):
        v = np.zeros((6, 1))
        v[0, 0] = h[0, i] * h[0, j]
        v[1, 0] = h[0, i] * h[1, j] + h[1, i] * h[0, j]
        v[2, 0] = h[1, i] * h[1, j]
        v[3, 0] = h[2, i] * h[0, j] + h[0, i] * h[2, j]
        v[4, 0] = h[2, i] * h[1, j] + h[1, i] * h[2, j]
        v[5, 0] = h[2, i] * h[2, j]
        return v

    V = np.zeros((2*n, 6))
    for i in range(n):
        V[i*2] = getv(Hs[i], 0, 1).T
        V[i*2+1] = (getv(Hs[i], 0, 0) - getv(Hs[i], 1, 1)).T

    # The solution of Vb = 0 is known as the eigenvetor of V^TV associated with the smallest eigenvalue
    eigenvalue, eigenvec = np.linalg.eig(np.dot(V.T, V))
    least_eigen_idx = np.argmin(eigenvalue)
    b = eigenvec[:, least_eigen_idx]

    B = np.zeros((3, 3))

    # B is symmetric,  b=[B11,B12,B22,B13,B23,B33]T .
    B[0, 0] = b[0]
    B[0, 1] = B[1, 0] = b[1]
    B[1, 1] = b[2]
    B[0, 2] = B[2, 0] = b[3]
    B[1, 2] = B[2, 1] = b[4]
    B[2, 2] = b[5]

    return B


def extract_intrinsic(B):
    """
    Implementation of Appendix B---Extraction of the Intrisic Parameters from matrix B
    :param B: matrix B is denoted by B = A^(-T)A^(-1)
    :return: camera intrinsic matrix A
    """

    v0 = (B[0, 1]*B[0, 2] - B[0, 0]*B[1, 2])/(B[0, 0]*B[1, 1] - B[0, 1]**2)
    Lambda = B[2, 2] - (B[0, 2]**2 + v0*(B[0, 1]*B[0, 2] - B[0, 0]*B[1, 2]))/B[0, 0]
    alpha = np.sqrt(Lambda/B[0, 0])
    beta = np.sqrt(Lambda*B[0, 0]/(B[0, 0]*B[1, 1] - B[0, 1]**2))
    gamma = - B[0, 1]*(alpha**2)*beta/Lambda
    u0 = gamma*v0/beta - B[0, 2]*(alpha**2)/gamma

    A = np.zeros((3, 3))
    A[0, 0] = alpha
    A[0, 1] = gamma
    A[0, 2] = u0
    A[1, 1] = beta
    A[1, 2] = v0
    A[2, 2] = 1

    return A


def compute_extrinsic(A, H):
    """
    Once A is known, the extrinsic parameters for each image is readily computed.
    But because of noise in data, the so-computed matrix does not in general satisfy the properties of a rotation matrix.
    :param A: Intrinsic matrix
    :param H: Homography for one picture
    :return:
    """
    Lambda_recip = np.sum(np.mat(A).I * np.mat(H[:, 0]).T)

    R = np.zeros((3, 3))
    R[:, 0] = np.squeeze(np.mat(A).I * np.mat(H[:, 0]).T)/ Lambda_recip
    R[:, 1] = np.squeeze(np.mat(A).I * np.mat(H[:, 1]).T) / Lambda_recip
    R[0, 2] = R[0, 1]*R[1, 2] - R[0, 2]*R[1, 1]
    R[1, 2] = R[0, 2]*R[1, 0] - R[0, 0]*R[1, 2]
    R[2, 2] = R[0, 0]*R[1, 1] - R[0, 1]*R[1, 0]

    t = (np.mat(A).I * np.mat(H[:, 2]).T) / Lambda_recip
    t = np.squeeze(t)
    return R, t

def img_coordinates(A, R, t, objpoints):
    """
    From object points in real world to compute theoretical 2D image points
    :param A: Intrinsic matrix
    :param R: Rotation matrix
    :param t: Translation vector
    :param objpoints: 3D coordinate points in real world
    :return: theoretical points of image pixel and normal coordinates
    """
    n = objpoints.shape[0]

    # object points in homogenous coordinates
    obj_homo = np.zeros((n, 4))
    obj_homo[:, :2] = objpoints[:, :2]
    obj_homo[:, 3] = objpoints[:, 2]

    Rt = np.zeros((3, 4))
    Rt[:, :3] = R
    Rt[:, 3] = t
    # P = A*[R|t]
    P = np.mat(A) * np.mat(Rt)

    # image points on the image plane
    img_pixel = P * np.mat(obj_homo).T
    # convert to homogenous coordinates which the 3rd value equals one
    img_pixel[0] = img_pixel[0] / img_pixel[2]
    img_pixel[1] = img_pixel[1] / img_pixel[2]
    img_pixel = np.mat(img_pixel).T

    # image points in camera coordinates
    img_real = np.mat(np.mat(Rt) * np.mat(obj_homo).T)
    img_real[0] = img_real[0] / img_real[2]
    img_real[1] = img_real[1] / img_real[2]
    img_real = np.mat(img_real).T

    return img_pixel, img_real

def get_radialCoefficients(distorted_pixel_image, ideal_pixel_image, ideal_real_image, u0, v0):
    """
    Section 3.3 Estimating Radial Distortion by Alternation.
    Get the radial distortion coefficients-- k1, k2 -- only the first two terms
    since more elaborated modeling not only would not help, but also would cause numerical instability
    :param distorted_pixel_image: observed distorted pixel image coordinates
    :param ideal_pixel_image: the ideal (nonobservable distortion-free) pixel image coordinates
    :param ideal_real_image: the ideal (distortion-free) image coordinates.
    :param u0: x value of optical points
    :param v0: y value of optical points
    :return: [k1, k2]
    """

    n = len(distorted_pixel_image)
    m = distorted_pixel_image[0].shape[0]

    D = np.zeros((2*n*m, 2))
    d = np.zeros((2*n*m, 1))

    for i in range(n):
        for j in range(m):
            r = ideal_real_image[i][j, 0]**2 + ideal_real_image[i][j, 1]**2
            du = ideal_pixel_image[i][j, 0] - u0
            dv = ideal_pixel_image[i][j, 1] - v0
            D[2*(m*i + j), 0] = du * r
            D[2*(m*i + j), 1] = du * (r**2)
            D[2*(m*i + j)+1, 0] = dv * r
            D[2*(m*i + j)+1, 1] = dv * (r**2)
            d[2*(m*i + j)] = distorted_pixel_image[i][j, 0] - ideal_pixel_image[i][j, 0]
            d[2*(m*i + j)+1] = distorted_pixel_image[i][j, 1] - ideal_pixel_image[i][j, 1]


    k = (np.mat(D).T * np.mat(D)).I * np.mat(D).T * np.mat(d)
    return k

def calibrationImplement(objpoints, imgpoints):
    """
    According to the 3D object points and 2D image, get parameters of the camera
    :param objpoints: 3D object points
    :param imgpoints: 2D pixel image
    :return: camera matrix A, distortion coeffients k including only k1 and k2 for radial distortion,
    list of rotation matrix R, list of translation vector t
    """

    n = len(objpoints)

    Hs = []
    for i in range(n):
        h = estimate_homography(objpoints[i], imgpoints[i])
        Hs.append(h)

    B = getMatrixB(Hs)
    A = extract_intrinsic(B)
    u0 = A[0, 2]
    v0 = A[1, 2]

    Rs = []
    ts = []
    for i in range(n):
        R ,t = compute_extrinsic(A, Hs[i])
        Rs.append(R)
        ts.append(t)

    ideal_pixel_imgs = []
    ideal_real_imgs = []
    for i in range(n):
        ideal_pixel_img, ideal_real_img = img_coordinates(A, Rs[i], ts[i], objpoints[i])
        ideal_pixel_imgs.append(ideal_pixel_img)
        ideal_real_imgs.append(ideal_real_img)

    k = get_radialCoefficients(imgpoints, ideal_pixel_imgs, ideal_real_imgs, u0, v0)

    return A, k, Rs, ts

if __name__ == "__main__":
    objpoints, imgpoints, imageSize = calib.find_points()
    mtx, dist, rvecs, tvecs = calibrationImplement(objpoints, imgpoints)
    print("camera matrix is ")
    print(mtx)
    print("distortion coeffients is ")
    print(dist)
    print("rotation matrix is ")
    print(rvecs)
    print("translation matrix is ")
    print(tvecs)