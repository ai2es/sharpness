import numpy as np
from sharpness.metric_list import metric_f, single_metrics


# Function to chunk image into tiles to create sharpness heatmap
# metric can be a general callable, but if it is an element from metric_f, then whether it takes one or two inputs is set automatically.
def Heatmap(img1, img2, metric, block_size, pad_len, pad_mode='reflect', block_stride=None, bivariate=None):

    # Check whether we are dealing with univariate or bivariate function, and raise appropriate errors
    single_metric_f = {key: metric_f[key] for key in single_metrics}
    if bivariate is None:
        if metric in single_metric_f.values():
            bivariate = False
        elif metric in metric_f.values():
            bivariate = True
        else:
            raise ValueError('Metric is not in list of known metrics; must specify number of inputs with "bivariate" option')

    if bivariate and (img2 is None):
        raise ValueError('Metric requires two inputs; only one is given')

    if bivariate and (img1.shape != img2.shape):
        raise ValueError(f'For bivariate metrics, images must be of the same shape. Got {img1.shape} and {img2.shape}')

    # Do some initial setup
    if block_stride is None:
        block_stride = block_size//4
        # can't have block_stride be less than 2
        if block_stride <= 1:
            block_stride = 2
    if pad_mode is not None:
        img1 = np.pad(img1, pad_len, mode=pad_mode)
        if img2 is not None:
            img2 = np.pad(img2, pad_len, mode=pad_mode)

    # Main loop, which depends on the type of data we used
    if bivariate:  # Bivariate Case
        (num_rows, num_cols) = img1.shape
        res = np.zeros(img1.shape) - 100

        # Run the main loop, checking only valid blocks and striding by block_stride
        for row in range(block_size//2, num_rows - block_size//2 + 1, block_stride):
            for col in range(block_size//2, num_cols - block_size//2 + 1, block_stride):

                # Subset the block of interest
                block1 = img1[row - block_size//2:row + block_size//2, col - block_size//2:col + block_size//2]
                block2 = img2[row - block_size//2:row + block_size//2, col - block_size//2:col + block_size//2]

                # If there is sufficient contrast, compute spectral slope and apply logistic function; else, return 0
                val = metric(block1, block2)

                # Fill result in the appropriate region with the value for this block
                res[row - block_stride//2:row + block_stride//2, col - block_stride//2:col + block_stride//2] = val

        # Crop to remove padding
        if pad_mode is not None:
            res = res[pad_len:num_rows - pad_len, pad_len:num_cols - pad_len]

    else:  # Univariate Case
        if img2 is None:  # Only one input case
            (num_rows, num_cols) = img1.shape
            res = np.zeros(img1.shape) - 100

            # Run the main loop, checking only valid blocks and striding by block_stride
            for row in range(block_size//2, num_rows - block_size//2 + 1, block_stride):
                for col in range(block_size//2, num_cols - block_size//2 + 1, block_stride):

                    # Subset the block of interest
                    block1 = img1[row - block_size//2:row + block_size//2, col - block_size//2:col + block_size//2]

                    # If there is sufficient contrast, compute spectral slope and apply logistic function; else, return 0
                    val = metric(block1)

                    # Fill result in the appropriate region with the value for this block
                    res[row - block_stride//2:row + block_stride//2, col - block_stride//2:col + block_stride//2] = val

            # Crop to remove padding
            if pad_mode is not None:
                res = res[pad_len:num_rows - pad_len, pad_len:num_cols - pad_len]

        else:  # Two input case
            imgs = [img1, img2]
            res = []
            for img in imgs:
                (num_rows, num_cols) = img.shape
                temp_res = np.zeros(img.shape) - 100

                # Run the main loop, checking only valid blocks and striding by block_stride
                for row in range(block_size//2, num_rows - block_size//2 + 1, block_stride):
                    for col in range(block_size//2, num_cols - block_size//2 + 1, block_stride):

                        # Subset the block of interest
                        block = img[row - block_size//2:row + block_size//2, col - block_size//2:col + block_size//2]

                        # If there is sufficient contrast, compute spectral slope and apply logistic function; else, return 0
                        val = metric(block)

                        # Fill result in the appropriate region with the value for this block
                        temp_res[row - block_stride//2:row + block_stride//2, col - block_stride//2:col + block_stride//2] = val

                # Crop to remove padding
                if pad_mode is not None:
                    temp_res = temp_res[pad_len:num_rows - pad_len, pad_len:num_cols - pad_len]

                res.append(temp_res)

    if np.any(np.isnan(res)):
        if type(res) is list:
            print(f'NaNs encountered in {metric.__name__}:\n'
                  f'{np.sum(np.isnan(res[0]))} NaNs out of {np.prod(res[0].shape)} total blocks in image 0\n'
                  f'{np.sum(np.isnan(res[1]))} NaNs out of {np.prod(res[1].shape)} total blocks in image 1')
        else:
            print(f'NaNs encountered in {metric.__name__}: {np.sum(np.isnan(res))} NaNs out of {np.prod(res.shape)} total blocks')

    return res


# Function intended to work in "compute_all_metrics" that takes in a list of metric names, and outputs a dictionary of heatmaps
def heatmap_list(img1, img2, metrics, block_size, pad_len, pad_mode='reflect', block_stride=None):
    # Check that the syntax is correct
    if not set(metrics).issubset(set(metric_f.keys())):
        raise ValueError(f'Metric(s) {set(metrics) - set(metric_f.keys())} is not known.')

    if img1.shape != img2.shape:
        raise ValueError(f'Images must be of the same shape. Got {img1.shape} and {img2.shape}')

    biv_dict = {}
    for metric in metrics:
        biv_dict[metric] = False if metric in single_metrics else True

    # Do some initial setup
    if block_stride is None:
        block_stride = block_size//4
        # can't have block_stride be less than 2
        if block_stride <= 1:
            block_stride = 2
    if pad_mode is not None:
        img1 = np.pad(img1, pad_len, mode=pad_mode)
        img2 = np.pad(img2, pad_len, mode=pad_mode)

    (num_rows, num_cols) = img1.shape
    res = {}
    for metric in metrics:
        if metric not in single_metrics:
            res[metric] = np.zeros(img1.shape) - 100
        else:
            res[metric] = [np.zeros(img1.shape) - 100, np.zeros(img1.shape) - 100]

    # Run the main loop, checking only valid blocks and striding by block_stride
    for row in range(block_size//2, num_rows - block_size//2 + 1, block_stride):
        for col in range(block_size//2, num_cols - block_size//2 + 1, block_stride):

            # Subset the block of interest
            block1 = img1[row - block_size//2:row + block_size//2, col - block_size//2:col + block_size//2]
            block2 = img2[row - block_size//2:row + block_size//2, col - block_size//2:col + block_size//2]

            # If there is sufficient contrast, compute spectral slope and apply logistic function; else, return 0
            for metric in metrics:
                f = metric_f[metric]
                if metric not in single_metrics:
                    val = f(block1, block2)
                    res[metric][row - block_stride//2:row + block_stride//2, col - block_stride//2:col + block_stride//2] = val
                else:
                    val = (f(block1), f(block2))
                    res[metric][0][row - block_stride//2:row + block_stride//2, col - block_stride//2:col + block_stride//2] = val[0]
                    res[metric][1][row - block_stride//2:row + block_stride//2, col - block_stride//2:col + block_stride//2] = val[1]

    # Crop to remove padding
    if pad_mode is not None:
        for metric in metrics:
            if metric not in single_metrics:
                res[metric] = res[metric][pad_len:num_rows - pad_len, pad_len:num_cols - pad_len]
            else:
                res[metric][0] = res[metric][0][pad_len:num_rows - pad_len, pad_len:num_cols - pad_len]
                res[metric][1] = res[metric][1][pad_len:num_rows - pad_len, pad_len:num_cols - pad_len]

    for metric in metrics:
        if np.any(np.isnan(res[metric])):
            print(f'NaNs encountered in {metric}; {np.sum(np.isnan(res[metric]))} NaNs out of {np.prod(res[metric].shape)} total blocks')

    return res