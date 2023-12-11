import numpy as np
from .metrics import (
    mse,
    mae,
    rmse,
    grad,
    s1,
    total_variation
)
from .gradient import (
    psnr,
    normalized_cross_correlation,
    gradient_difference_similarity,
    gradient_rmse,
    laplacian_rmse,
    histogram_intersection,
    hog_pearson,
    grad_total_variation
)
from .fourier import (
    fourier_rmse,
    fourier_total_variation
)
from .wavelet import (
    wavelet_image_similarity,
    wavelet_total_variation
)

metric_f = {
    'mse': mse,
    'mae': mae,
    'rmse': rmse,
    's1': s1,
    "psnr": psnr,
    "ncc": normalized_cross_correlation,
    'grad': grad,
    "grad-ds": gradient_difference_similarity,
    "grad-rmse": gradient_rmse,
    "laplace-rmse": laplacian_rmse,
    "fourier-rmse": fourier_rmse,
    "hist-int": histogram_intersection,
    #"gpd": gradient_profile_difference,
    "hog-pearson": hog_pearson,
    "wavelet-similarity": wavelet_image_similarity,
    "tv": total_variation,
    "grad-tv": grad_total_variation,
    "fourier-tv": fourier_total_variation,
    "wavelet-tv": wavelet_total_variation
}

single_metrics = ["grad", "s1", "tv", "grad-tv", "fourier-tv", "wavelet-tv"]


def compute_all_metrics(X, T) -> dict:
    """Check that X and T are 2-dim arrays"""
    assert np.ndim(X)==2, f'Input must be 2-dimensional; got {np.ndim(X)} dimensions for X'
    if T is not None:
        assert np.ndim(T)==2, f'Input must be 2-dimensional; got {np.ndim(T)} dimensions for T'

    """Compute all evaluation metrics."""
    results = dict()
    metrics_to_compute = single_metrics if T is None else metric_f.keys()

    for metric in metrics_to_compute:
        f = metric_f.get(metric)
        if f is not None:
            try:
                if T is not None:
                    if metric in single_metrics:
                        results[metric] = f(X), f(T)
                    else:
                        results[metric] = f(X, T)
                else:
                    results[metric] = f(X)
            except Exception as e:
                print(f'Failed to compute {metric}: {e}')
        else:
            print(f'Unknown metric name: {metric}')

    return results


def compute_metric(X, T, metric: str):
    """Check that X and T are 2-dim arrays"""
    assert np.ndim(X)==2; f'Input must be 2-dimensional; got {np.ndim(X)} dimensions for X'
    if T is not None:
        assert np.ndim(T)==2; f'Input must be 2-dimensional; got {np.ndim(T)} dimensions for T'

    """Compute specified evaluation metric"""
    f = metric_f.get(metric)
    if f is None:
        raise ValueError(f'Unknown metric name: {metric}')

    if T is not None:
        if metric in single_metrics:
            return f(X), f(T)
        else:
            return f(X, T)
    else:
        return f(X)
