#!/usr/bin/env python

import os
import numpy as np
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import imageio
import matplotlib.pyplot as plt
from pylbfgs import owlqn


# Evaluation method (always use #1)
# 1: in-memory dct2 version (fast, efficient for all size images)
# 2: in-memory kron version (fast, but only for images smaller than 100x100)
# 3: file-based kron version (experimental, slow, lots of disk
#    space required [Order(rows x cols x samples)])
EVAL_METHOD = 1

# Fraction to scale the original image
SCALE = 0.5

# Fraction of the scaled image to randomly sample
SAMPLE = 0.2

# Coeefficient for the L1 norm of variables (see OWL-QN algorithm)
ORTHANTWISE_C = 5

# File paths
ORIG_IMAGE_PATH = 'test/testimage.png'
A_FILE_PATH = 'test/a_matrix.npy'
B_FILE_PATH = 'test/b_vector.npy'


def dct2(x):
    """Return 2D discrete cosine transform.
    """
    return spfft.dct(
        spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def idct2(x):
    """Return inverse 2D discrete cosine transform.
    """
    return spfft.idct(
        spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def kron_rows(A, B, I, f=None):
    """Return individual rows of K=kron(A,B) if `f` is None. Otherwise
    save the matrix to file.
    """

    # find row indices of A and B
    ma, na = A.shape
    mb, nb = B.shape
    R = np.floor(I / mb).astype('int')  # A row indices of interest
    S = np.mod(I, mb)  # B row indices of interest

    # calculate kronecker product rows
    n = na * nb
    if f is None:
        K = np.zeros((I.size, n))

    for j, (r, s) in enumerate(zip(R, S)):
        row = np.multiply(
            np.kron(A[r, :], np.ones((1, nb))),
            np.kron(np.ones((1, na)), B[s, :])
            )
        if f is None:
            K[j, :] = row
        else:
            row.tofile(f)

    if f is None:
        return K


def progress(x, g, fx, xnorm, gnorm, step, k, ls):
    """Just display the current iteration.
    """
    print('Iteration {}'.format(k))
    return 0


_image_dims = None  # track target image dimensions here
_ri_vector = None  # reference the random sampling indices here
_b_vector = None  # reference the sampled vector b here


def evaluate(x, g, step):
    """An in-memory evaluation callback.
    """

    # we want to return two things:
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((_image_dims[1], _image_dims[0])).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)

    # stack columns and extract samples
    Ax = Ax2.T.flat[_ri_vector].reshape(_b_vector.shape)

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - _b_vector
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[_ri_vector] = Axb  # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    AtAxb = AtAxb2.T.reshape(x.shape)  # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx


_A_matrix = None  # reference the dct matrix operator A here


def evaluate_kron(x, g, step):
    """An in-memory evaluation callback using the kronecker product.
    """

    # calculate the 2-norm squared of the residual vector
    p = np.dot(_A_matrix, x)
    fx = np.sum(np.power(_b_vector - p, 2))

    # calculate the gradient vector
    atax = np.dot(_A_matrix.T, p)
    atb = np.dot(_A_matrix.T, _b_vector)
    np.copyto(g, 2 * (atax - atb))

    return fx


def evaluate_kron_from_file(x, g, step):
    """A slower, but more memory efficient, evaluation callback
    using the kronecker product.
    """

    # read in b data from file
    with open(B_FILE_PATH, 'rb') as f:
        b = np.fromfile(f)

    # allocate vectors
    k = b.size
    n = x.size
    Ax = np.zeros(b.shape)
    Atax = np.zeros(x.shape)
    Atb = np.zeros(x.shape)

    # read in A data from file
    with open(A_FILE_PATH, 'rb') as f:
        for i in range(k):
            row = np.fromfile(f, count=n)
            Ax[i] = np.dot(row, x)
            for j in range(n):
                Atax[j] += row[j] * Ax[i]
                Atb[j] += row[j] * b[i]

    # calculate the 2-norm squared of the residual vector
    fx = np.sum(np.power(b - Ax, 2))

    # calculate the gradient vector
    np.copyto(g, 2 * (Atax - Atb))

    return fx


def main():

    global _b_vector, _A_matrix, _image_dims, _ri_vector

    # read image in, then convert to grayscale and then downscale it
    Xorig = imageio.imread(ORIG_IMAGE_PATH)
    # Cast to float before grayscale conversion
    Xorig = Xorig.astype(float)
    # Preserving the behaviour of mode='L' from the deprecated scipy.ndimage.imread
    Xorig = Xorig[:, :, 0] * 299/1000 + Xorig[:, :, 1] * 587/1000 + Xorig[:, :, 2] * 114/1000 
    X = spimg.zoom(Xorig, SCALE)
    ny, nx = X.shape

    # take random samples of image, store them in a vector b
    k = round(nx * ny * SAMPLE)
    ri = np.random.choice(nx*ny, k, replace=False)  # random sample of indices
    b = X.T.flat[ri].astype(float)  # important: cast to 64 bit

    if EVAL_METHOD == 1:

        # This method evaluates the objective function sum((Ax-b).^2) and its
        # gradient without ever actually generating A (which can be massive).
        # Our ability to do this stems from our knowledge that Ax is just the
        # sampled idct2 of the spectral image (x in matrix form).

        # save image dims, sampling vector, and b vector and to global vars
        _image_dims = (ny, nx)
        _ri_vector = ri
        _b_vector = np.expand_dims(b, axis=1)

        # perform the L1 minimization in memory
        Xat2 = owlqn(nx*ny, evaluate, progress, ORTHANTWISE_C)

    elif EVAL_METHOD == 2:

        # This method computes a dct2 matrix operator A (a kronecker product)
        # and uses it directly to evaluate the objective function. However,
        # as the size of the target image gets bigger, A gets unmanageable.

        # save refs to global vars
        _b_vector = np.expand_dims(b, axis=1)
        _A_matrix = kron_rows(
            spfft.idct(np.identity(nx), norm='ortho', axis=0),
            spfft.idct(np.identity(ny), norm='ortho', axis=0),
            ri
            )

        # perform the L1 minimization in memory
        Xat2 = owlqn(nx*ny, evaluate_kron, progress, ORTHANTWISE_C)

    elif EVAL_METHOD == 3:

        # This method computes a dct2 matrix operator A (a kronecker product),
        # potentially very large, and saves it to file. The evaluation method
        # then loads the data from file each time it is called to perform the
        # calculations. This method is very memory efficient, however the
        # kronecker matrix grows on an order of (nx*ny)^2, so the file can
        # become massive. I/O operations also slow this method down
        # considerably.
        # Don't use this method -- it is here mainly for education.

        # save the b samples to file
        with open(B_FILE_PATH, 'wb') as f:
            print('Writing {} values to {}'.format(k, B_FILE_PATH))
            b.tofile(f)

        # create dct matrix operator A, saving it directly to file
        try:
            os.remove(A_FILE_PATH)
        except:
            pass
        with open(A_FILE_PATH, 'w+b') as f:
            print('Writing {} values to {}'.format(nx * ny * k, A_FILE_PATH))
            kron_rows(
                spfft.idct(np.identity(nx), norm='ortho', axis=0),
                spfft.idct(np.identity(ny), norm='ortho', axis=0),
                ri, f
                )

        # perform the L1 minimization from file
        Xat2 = owlqn(nx*ny, evaluate_kron_from_file, progress, ORTHANTWISE_C)

    # transform the output back into the spatial domain
    Xat = Xat2.reshape(nx, ny).T  # stack columns
    Xa = idct2(Xat)

    # create images of mask (for visualization)
    mask = np.zeros(X.shape)
    mask.T.flat[ri] = 255
    Xm = 255 * np.ones(X.shape)
    Xm.T.flat[ri] = X.T.flat[ri]

    # display the result
    f, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(X, cmap='hot', interpolation='none',vmin=0, vmax=255)
    ax[1].imshow(Xm, cmap='hot', interpolation='none',vmin=0, vmax=255)
    ax[2].imshow(Xa, cmap='hot', interpolation='none',vmin=0, vmax=255)
    plt.show()


if __name__ == '__main__':
    main()
