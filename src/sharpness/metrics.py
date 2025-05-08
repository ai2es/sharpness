"""
Module containing low-level metric functions

The dictionary "metric_f" contains the metric functions as items, with their short names
being the keys. All these functions are callable in the format func(X, T), where X and T
are two images whose sharpness is to be compared.

The list "single_metrics" contains the keys for those functions in "metric_f" that can
be evaluated on single images -- i.e., they can be used in the format func(X), where X
is a single image.
"""

import numpy as np
from numpy.polynomial import Polynomial as P
from skimage.metrics import structural_similarity
import cv2
import pywt
import scipy.ndimage as nd
from functools import partial


def rmse(X, T):
    """Bivariate -- Root Mean Squared Error"""
    return np.sqrt(np.mean((X - T) ** 2))


def ssim(X, T, win_size=7, data_range=255):
    """Bivariate -- SSIM from scikit-image"""
    return structural_similarity(X, T, win_size=win_size, data_range=data_range)


def total_variation(X):
    """Univariate -- Total variation of an image"""
    horizontal_tv = np.sum(np.abs(X[:, :-1] - X[:, 1:]))
    vertical_tv = np.sum(np.abs(X[:-1, :] - X[1:, :]))
    tv = horizontal_tv + vertical_tv
    return tv


def mean_gradient_magnitude(image):
    """Univariate -- computes the mean of the gradient magnitude map"""
    # Ensure the image is a NumPy array with float data type
    image = image.astype(float)

    # Calculate gradients of the image -- border handling is with reflect_101 method.
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Compute the Mean Gradient Magnitude
    mgm = np.mean(gradient_magnitude)

    return mgm


def grad_total_variation(image):
    """Univariate -- computes total variation of the gradient magnitude map"""
    # Ensure the image is a NumPy array with float data type
    image = image.astype(float)

    # Calculate the horizontal and vertical gradients using central differences
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the total variation as the L1 norm of the gradients
    tv = np.sum(np.abs(gradient_x)) + np.sum(np.abs(gradient_y))

    return tv


def gradient_rmse(image1, image2):
    """Bivariate -- Computes the RMSE between gradient magnitude maps of two images"""
    image1 = image1.astype(float)
    image2 = image2.astype(float)

    gradient_x1 = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y1 = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude1 = np.sqrt(gradient_x1**2 + gradient_y1**2)

    gradient_x2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude2 = np.sqrt(gradient_x2**2 + gradient_y2**2)

    difference = np.abs(gradient_magnitude1 - gradient_magnitude2)
    rmse = np.sqrt(np.mean(difference**2))
    return rmse


def laplacian_rmse(image1, image2):
    """Bivariate -- Computes the RMSE between Laplacian maps of two images"""
    image1 = image1.astype(float)
    image2 = image2.astype(float)

    # Compute Laplacian images
    laplacian1 = cv2.Laplacian(image1, cv2.CV_64F)
    laplacian2 = cv2.Laplacian(image2, cv2.CV_64F)

    # Calculate pixel-wise differences
    difference = np.abs(laplacian1 - laplacian2)

    # Calculate Mean Squared Error
    rmse = np.sqrt(np.mean(difference**2))

    return rmse


def compute_power_spectrum(image, hanning=True):
    """Utility method used for Fourier RMSE"""
    N = image.shape[0]
    if hanning:
        # Set up 2D Hanning window to deal with edge effects
        window = np.hanning(N)
        window = np.outer(window, window)
        image = image * window

    # Compute the power spectrum of an image
    f_transform = np.fft.fft2(image)
    power_spectrum = np.abs(f_transform) ** 2
    return power_spectrum


def fourier_rmse(image1, image2, hanning=True):
    """Bivariate -- Unweighted RMSE between power spectra of the input images"""
    # Compute power spectra of both images
    power_spectrum1 = compute_power_spectrum(image1, hanning=hanning)
    power_spectrum2 = compute_power_spectrum(image2, hanning=hanning)

    # Compute the mean squared error between power spectra
    mse = np.mean((power_spectrum1 - power_spectrum2) ** 2)
    return np.sqrt(mse)


def fourier_total_variation(image, hanning=True):
    """Univariate -- Total variation within the power spectra of a given image"""
    N = image.shape[0]
    if hanning:
        # Set up 2D Hanning window to deal with edge effects
        window = np.hanning(N)
        window = np.outer(window, window)
        image = image * window

    f_transform = np.fft.fft2(image)
    tv = np.sum(np.abs(f_transform))
    return tv


def wavelet_total_variation(image, wavelet="haar", level=1):
    """Univariate -- metric to compute total variation using wavelet coefficients"""
    # Ensure the image is a NumPy array with float data type
    image = image.astype(float)

    # Apply wavelet transform
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # Calculate the Wavelet Total Variation
    wavelet_tv = 0
    for c in coeffs:
        wavelet_tv += np.sum(np.abs(c))

    return wavelet_tv


def polar_average(spectrum_2d, num_angles=360):
    """
    Utility function to compute polar average of 2D FFT

    args:
        spectrum_2d (np.array): Output of 2D FFT, either in magnitude or complex form.
        num_angles (int): How many angles to average across

    returns:
        f (np.array): 1D array of frequency radii
        s (np.array): 1D array of polar averaged values at the corresponding frequency
            radii from f.
    """
    N = spectrum_2d.shape[0]

    # Set the FFT mean to the average of the first two low-frequency values
    # This avoids throwing off the interpolated values
    spectrum_2d[0, 0] = np.mean([spectrum_2d[0, 1], spectrum_2d[1, 0]])

    # Generate grid coordinates in terms of polar coordinates
    # excluding the global average but including the Nyquist frequency at N//2.
    xs = []
    ys = []
    thetas = np.linspace(0, 2 * np.pi, num_angles + 1)[:-1]
    for r in range(1, N // 2 + 1):
        xs.append(r * np.cos(thetas))
        ys.append(r * np.sin(thetas))
    grid_coords = np.array([np.concatenate(xs), np.concatenate(ys)])

    # Interpolate values at those coordinates
    s_full = nd.map_coordinates(spectrum_2d, grid_coords, mode="grid-wrap", order=1)
    s_full = s_full.reshape(-1, num_angles)

    # Average together
    s = s_full.mean(axis=1)

    # Generate frequency coordinates
    f = np.linspace(0, 0.5, s.shape[0] + 1)

    # Exclude 0th frequency, as we didn't compute an s value for that
    f = f[1:]

    return f, s


def spec_slope(image, hanning=True):
    """Univariate -- Computes raw spectral slope metric"""
    N = image.shape[0]
    if hanning:
        # Set up 2D Hanning window to deal with edge effects
        window = np.hanning(N)
        window = np.outer(window, window)
        image = image * window

    # Compute polar averaged spectral values
    # f is the frequency radius
    # s is the average value for that frequency
    [f, s] = polar_average(np.abs(np.fft.fft2(image)))

    # Fit a line to the log-log transformed data
    line = P.fit(np.log(f), np.log(s), 1)
    res = line.coef[1]
    return res


def s1(
    image,
    contrast_threshold=None,
    brightness_threshold=None,
    brightness_mult=False,
    hanning=True,
):
    """Univariate -- Computes S1 metric from Vu et al (2009)"""

    if (contrast_threshold is not None) and (
        np.nanmax(image) - np.nanmin(image) < contrast_threshold
    ):
        val = np.nan
    elif (brightness_threshold is not None) and (
        np.nanmax(image) < brightness_threshold
    ):
        val = np.nan
    else:
        if brightness_mult:
            val = spec_slope(image, hanning) * np.nanmean(image)
        else:
            val = spec_slope(image, hanning)

    return val


metric_f = {
    "rmse": rmse,
    "ssim": partial(ssim, data_range=255),
    "tv": total_variation,
    "grad-mag": mean_gradient_magnitude,
    "grad-tv": grad_total_variation,
    "grad-rmse": gradient_rmse,
    "laplace-rmse": laplacian_rmse,
    "fourier-rmse": fourier_rmse,
    "fourier-tv": fourier_total_variation,
    "spec-slope": spec_slope,
    "s1": partial(
        s1, contrast_threshold=5, brightness_threshold=20, brightness_mult=False
    ),
    "wavelet-tv": wavelet_total_variation,
}
metric_f["ssim"].__name__ = "ssim"
metric_f["s1"].__name__ = "s1"

single_metrics = [
    "grad-mag",
    "spec-slope",
    "s1",
    "tv",
    "grad-tv",
    "fourier-tv",
    "wavelet-tv",
]
