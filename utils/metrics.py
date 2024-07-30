import cv2
import numpy as np
from skimage.feature import hog
from scipy.stats import pearsonr
import pywt
from numpy.polynomial import Polynomial as P
import scipy.ndimage as nd
from skimage.metrics import structural_similarity


def mse(X, T):
    """Mean Squared Error"""
    return np.mean((X - T) ** 2)


def mae(X, T):
    """Mean Absolute Error"""
    return np.mean(np.abs((X - T)))


def rmse(X, T):
    """Root Mean Squared Error"""
    return np.sqrt(mse(X, T))


def ssim(X, T, win_size=7, data_range=255):
    """SSIM from scikit-image"""
    return structural_similarity(X, T, win_size=win_size, data_range=data_range)


def total_variation(X):
    """ Total variation of an image """
    horizontal_tv = np.sum(np.abs(X[:, :-1] - X[:, 1:]))
    vertical_tv = np.sum(np.abs(X[:-1, :] - X[1:, :]))
    tv = horizontal_tv + vertical_tv
    return tv

def compute_power_spectrum(image, hanning=True):
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
    # Compute power spectra of both images
    power_spectrum1 = compute_power_spectrum(image1, hanning=hanning)
    power_spectrum2 = compute_power_spectrum(image2, hanning=hanning)

    # Compute the mean squared error between power spectra
    mse = np.mean((power_spectrum1-power_spectrum2)**2)
    return np.sqrt(mse)


def fourier_total_variation(image, hanning=True):
    N = image.shape[0]
    if hanning:
        # Set up 2D Hanning window to deal with edge effects
        window = np.hanning(N)
        window = np.outer(window, window)
        image = image * window

    f_transform = np.fft.fft2(image)
    tv = np.sum(np.abs(f_transform))
    return tv

# Function to compute S1 metric from Vu et al. paper
def s1(img, contrast_threshold=None, brightness_threshold=None, brightness_mult=False, hanning=True):

    if (contrast_threshold is not None) and (np.nanmax(img) - np.nanmin(img) < contrast_threshold):
        val = np.nan
    elif (brightness_threshold is not None) and (np.nanmax(img) < brightness_threshold):
        val = np.nan
    else:
        if brightness_mult:
            val = spec_slope(img, hanning) * np.nanmean(img)
        else:
            val = spec_slope(img, hanning)

    return val


# Compute the spectral slope
def spec_slope(block, hanning=True):
    N = block.shape[0]
    if hanning:
        # Set up 2D Hanning window to deal with edge effects
        window = np.hanning(N)
        window = np.outer(window, window)
        block = block * window

    # Compute polar averaged spectral values
    # f is the frequency radius
    # s is the average value for that frequency
    [f, s] = polar_average(np.abs(np.fft.fft2(block)))

    # Fit a line to the log-log transformed data
    line = P.fit(np.log(f), np.log(s), 1)
    res = line.coef[1]
    return res


# Given output of FFT, compute polar averaged version
# Returns a tuple (f, a)
# f is a 1D array of frequency radii
# s is a 1D array of the same length with the polar averaged value for the corresponding radii
def polar_average(spect, num_angles=360):
    N = spect.shape[0]

    spect[0, 0] = np.mean([spect[0, 1], spect[1, 0]])

    # Generate grid coordinates in terms of polar coordinates, excluding the global average but including the Nyquist frequency at N//2.
    xs = []
    ys = []
    thetas = np.linspace(0, 2*np.pi, num_angles+1)[:-1]
    for r in range(1, N//2+1):
        xs.append(r * np.cos(thetas))
        ys.append(r * np.sin(thetas))
    grid_coords = np.array([np.concatenate(xs), np.concatenate(ys)])

    # Obtain values at those coordinates
    s_full = nd.map_coordinates(spect, grid_coords, mode='grid-wrap', order=1)
    s_full = s_full.reshape(-1, num_angles)

    # Average together
    s = s_full.mean(axis=1)

    # Generate frequency coordinates
    f = np.linspace(0, 0.5, s.shape[0] + 1)

    # Exclude 0th frequency, as we didn't compute an s value for that
    f = f[1:]

    return f, s

def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    max_pixel_value = np.max(image1)
    eps = np.finfo(np.float32).tiny
    psnr_value = 20 * np.log10(max_pixel_value / (eps + np.sqrt(mse)))
    return psnr_value


def normalized_cross_correlation(image1, image2):
    ncc = np.sum(image1 * image2) / (np.sqrt(np.sum(image1 ** 2)) * np.sqrt(np.sum(image2 ** 2)))
    return ncc


def histogram_intersection(image1, image2, bins=256):
    cmin = np.nanmin([image1, image2])
    cmax = np.nanmax([image1, image2])
    hist1, _ = np.histogram(image1.flatten(), bins=bins, range=[cmin, cmax])
    hist2, _ = np.histogram(image2.flatten(), bins=bins, range=[cmin, cmax])
    intersection = np.minimum(hist1, hist2).sum() / np.maximum(hist1, hist2).sum()
    return intersection


def gradient_difference_similarity(image1, image2):
    # Ensure the image is a NumPy array with float data type
    image1 = image1.astype(float)
    image2 = image2.astype(float)

    gradient_x1 = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y1 = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude1 = np.sqrt(gradient_x1 ** 2 + gradient_y1 ** 2)

    gradient_x2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude2 = np.sqrt(gradient_x2 ** 2 + gradient_y2 ** 2)

    gds = np.sum(np.abs(gradient_magnitude1 - gradient_magnitude2)) / np.sum(gradient_magnitude1 + gradient_magnitude2)
    return gds


def gradient_rmse(image1, image2):
    image1 = image1.astype(float)
    image2 = image2.astype(float)

    gradient_x1 = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y1 = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude1 = np.sqrt(gradient_x1 ** 2 + gradient_y1 ** 2)

    gradient_x2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude2 = np.sqrt(gradient_x2 ** 2 + gradient_y2 ** 2)

    difference = np.abs(gradient_magnitude1 - gradient_magnitude2)
    rmse = np.sqrt(np.mean(difference**2))
    return rmse


def laplacian_rmse(image1, image2):
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


def histogram_of_oriented_gradients(image):
    hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    return hog_features


def hog_pearson(image1, image2):
    hog_features_1 = histogram_of_oriented_gradients(image1)
    hog_features_2 = histogram_of_oriented_gradients(image2)

    #squared_diff = [(x - y) ** 2 for x, y in zip(hog_features_1, hog_features_2)]
    #distance = sum(squared_diff) ** 0.5

    # HOG features for two images (hog_features_1 and hog_features_2)
    return pearsonr(hog_features_1, hog_features_2)[0]


def mean_gradient_magnitude(image):
    # Ensure the image is a NumPy array with float data type
    image = image.astype(float)

    # Calculate gradients of the image
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Compute the Mean Gradient Magnitude
    mgm = np.mean(gradient_magnitude)

    return mgm


def grad_total_variation(image):
    # Ensure the image is a NumPy array with float data type
    image = image.astype(float)

    # Calculate the horizontal and vertical gradients using central differences
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the total variation as the L1 norm of the gradients
    tv = np.sum(np.abs(gradient_x)) + np.sum(np.abs(gradient_y))

    return tv


# Example usage:
if __name__ == "__main__":
    # Load an example image (make sure to replace with your own image)
    from skimage.data import camera
    image1 = camera()
    image2 = camera()

    #  Calculate psnr
    psnr_value = psnr(image1, image2)
    print("psnr:", psnr_value)

    # Calculate Normalized Cross-Correlation
    ncc_value = normalized_cross_correlation(image1, image2)
    print("NCC:", ncc_value)

    # Calculate Mean Gradient Magnitude
    mgm_value = mean_gradient_magnitude(image1)
    print("MGM:", mgm_value)
    
    # Calculate Gradient Difference Similarity
    gds_value = gradient_difference_similarity(image1, image2)
    print("GDS:", gds_value)
    
    # Calculate Gradient-MSE
    gmd_value = gradient_rmse(image1, image2)
    print("G-RMSE:", gmd_value)

    # Calculate Laplacian-MSE
    mse_lap = laplacian_rmse(image1, image2)
    print("RMSE-Laplace:", mse_lap)

    # Calculate Histogram Intersection
    hist_intersection = histogram_intersection(image1, image2)
    print("Histogram Intersection:", hist_intersection)
    
    # Calculate Gradient Profile Difference
    #gpd_value = gradient_profile_difference(image1, image2)
    #print("GPD:", gpd_value)

    # Calculate Histogram of Oriented Gradients (HOG) for the image
    hog = hog_pearson(image1, image2)
    print("HOG pearson", hog)


def compute_wavelet_energy(image, wavelet='haar', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    energy = 0
    for c in coeffs:
        energy += np.sum(np.abs(c) ** 2)
    return energy

def compute_wavelet_entropy(image, wavelet='haar', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    entropy = 0
    for c in coeffs:
        normalized_c = np.abs(c) / np.sum(np.abs(c))
        entropy += -np.sum(normalized_c * np.log2(normalized_c + np.finfo(float).eps))
    return entropy

def wavelet_image_similarity(image1, image2):
    energy1 = compute_wavelet_energy(image1)
    energy2 = compute_wavelet_energy(image2)
    entropy1 = compute_wavelet_entropy(image1)
    entropy2 = compute_wavelet_entropy(image2)

    # Calculate similarity scores based on energy and entropy
    energy_similarity = np.exp(-abs(energy1 - energy2))
    entropy_similarity = np.exp(-abs(entropy1 - entropy2))

    # A weighted combination of energy and entropy similarity can be used
    similarity_score = 0.5 * energy_similarity + 0.5 * entropy_similarity

    return similarity_score

def wavelet_total_variation(image, wavelet='haar', level=1):
    # Ensure the image is a NumPy array with float data type
    image = image.astype(float)

    # Apply wavelet transform
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # Calculate the Wavelet Total Variation
    wavelet_tv = 0
    for c in coeffs:
        wavelet_tv += np.sum(np.abs(c))

    return wavelet_tv