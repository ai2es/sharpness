import cv2
import numpy as np

def compute_power_spectrum(image):
    # Compute the power spectrum of an image
    f_transform = np.fft.fft2(image)
    power_spectrum = np.abs(f_transform) ** 2
    return power_spectrum

def fourier_rmse(image1, image2):
    # Compute power spectra of both images
    power_spectrum1 = compute_power_spectrum(image1)
    power_spectrum2 = compute_power_spectrum(image2)

    # Compute the mean squared error between power spectra
    mse = np.mean((power_spectrum1-power_spectrum2)**2)
    return np.sqrt(mse)

def fourier_total_variation(image):
    f_transform = np.fft.fft2(image)
    tv = np.sum(np.abs(f_transform))
    return tv


if __name__ == '__main__':
    from skimage.data import camera
    image1 = camera()
    image2 = camera()

    fourier_mse = fourier_mse(image1, image2)
    print("Fourier power spectrum MSE:", fourier_mse)
