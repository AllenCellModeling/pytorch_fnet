#!/usr/bin/env python
import sys
sys.path.append('.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fnet
import fnet.data
import os
import pdb

def get_mask_shell(distance, shape):
    # get boolean mask to select pixels of shell with specified distance from edge
    assert all(distance < shape[i]//2 for i in range(len(shape)))
    slices_bigger = tuple(slice(distance, shape[i] - distance) for i in range(len(shape)))
    slices_smaller = tuple(slice(distance + 1, shape[i] - (distance + 1)) for i in range(len(shape)))
    mask = np.zeros(shape, dtype=np.bool)
    mask[slices_bigger] = True
    mask[slices_smaller] = False
    return mask

def get_mask(distance, shape, dims):
    # get boolean mask to select pixels of shell with specified distance from edge
    if isinstance(dims, int):
        dims = [dims]
    assert all(i < len(shape) for i in dims)
    assert all(distance < shape[i]//2 for i in range(len(shape)))
    slices_bigger = tuple(slice(distance, shape[i] - distance) if i in dims else slice(None) for i in range(len(shape)))
    slices_smaller = tuple(slice(distance + 1, shape[i] - (distance + 1)) if i in dims else slice(None) for i in range(len(shape)))
    mask = np.zeros(shape, dtype=np.bool)
    mask[slices_bigger] = True
    mask[slices_smaller] = False
    return mask

def get_mask_z(distance, shape):
    mask = np.zeros(shape, dtype=np.bool)
    mask[(distance, shape[0] - 1 - distance ), :, :] = True
    return mask

def get_mask_y(distance, shape):
    mask = np.zeros(shape, dtype=np.bool)
    mask[:, (distance, shape[1] - 1 - distance ), :] = True
    return mask

def get_mask_x(distance, shape):
    mask = np.zeros(shape, dtype=np.bool)
    mask[:, :, (distance, shape[2] - 1 - distance)] = True
    return mask

def make_plot(path_save, means, stds, ylabel='squared error', legend=None):
    if not isinstance(means, list):
        means = [means]
    if not isinstance(stds, list):
        stds = [stds]
    
    dirname = os.path.dirname(path_save)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig, ax = plt.subplots()
    # ax.errorbar(x, means, yerr=stds)
    for ar_mean in means:
        x = np.arange(len(ar_mean))
        ax.errorbar(x, ar_mean)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('distance from edge (px)')
    if legend is not None:
        ax.legend(legend)
    fig.savefig(path_save)
    print('saved:', path_save)

def update_stats_arrays(chunks, dims, sums, sums_of_squares, counts):
    assert chunks.ndim == 4
    shape_chunk = chunks[0].shape
    n_distances = len(sums)
    for d in range(n_distances):
        mask = get_mask(d, shape_chunk, dims)
        for chunk in chunks:
            pixels = chunk[mask]
            sums[d] += np.sum(pixels)
            sums_of_squares[d] += np.sum(pixels**2)
            counts[d] += len(pixels)

def get_means_stds(sums, sums_of_squares, counts):
    means = sums/counts
    variances = (sums_of_squares - sums**2/counts)/counts
    stds = np.sqrt(variances)
    return means, stds

def test_plot():
    path_save = 'tmp/tests/outputs/fake_error.png'
    chunks = test_get_chunks()
    n_distances = 8

    dims = (1,2)
    sums = np.zeros(n_distances)
    sums_of_squares = np.zeros(n_distances)
    counts = np.zeros(n_distances)
    update_stats_arrays(chunks, dims, sums, sums_of_squares, counts)
    ar_mean, ar_std = get_means_stds(sums, sums_of_squares, counts)
    ar_mean_2 = ar_mean*1.2
    make_plot(path_save, [ar_mean, ar_mean_2], ar_std)

def test_get_chunks(n_chunks=5, shape=(32, 32, 32)):
    rng = np.random.RandomState(0)
    chunks = rng.randn(*((n_chunks,) + shape))
    return chunks

def test_get_mask():
    chunk = test_get_chunks(n_chunks=1)[0]

    distance = 3
    n_side = 32
    mask = get_mask(distance, chunk.shape, dims=(2, 1, 0))
    n_side_shell = n_side - 2*distance
    count_exp = 2*n_side_shell**2 + (n_side_shell - 1)*4*(n_side_shell - 2)
    count_got = np.count_nonzero(mask)
    print('DEBUG: non-zeros exp: {} | got: {}'.format(count_exp, count_got))
    assert count_exp == count_got

    distance = 5
    n_side_shell = n_side - 2*distance
    mask = get_mask(distance, chunk.shape, dims=(2, 1))
    count_exp = (n_side_shell - 1)*4*n_side
    count_got = np.count_nonzero(mask)
    print('DEBUG: non-zeros exp: {} | got: {}'.format(count_exp, count_got))
    assert count_exp == count_got
    
            
def test():
    n_distances = 5
    chunks = test_get_chunks()
    n_chunks = chunks.shape[0]
    shape_chunk = chunks.shape[1:]
    
    results = {}
    for idx_chunk, chunk in enumerate(chunks):
        for d in range(n_distances):
            mask = get_mask(d, shape)
            pixels_at_d = chunk[mask]
            if d not in results:
                results[d] = np.zeros(len(pixels_at_d)*n_chunks)
            all_pixels_at_d = results[d]
            offset = idx_chunk*len(pixels_at_d)
            all_pixels_at_d[offset: offset + len(pixels_at_d)] = pixels_at_d
    print(
        '*** total ***')
    for key, val in results.items():
        print('d: {} | mean: {} | var: {} | std: {}'.format(key, np.mean(val), np.var(val), np.std(val)))

    print('*** chek calc ***')
    sums = np.zeros(n_distances)
    sums_of_squares = np.zeros(n_distances)
    counts = np.zeros(n_distances)
    update_stats_arrays(chunks, sums, sums_of_squares, counts)
    
    means = sums/counts
    variances = (sums_of_squares - sums**2/counts)/counts
    stds = np.sqrt(variances)
    for d in range(n_distances):
        print('d: {} | mean: {} | var: {} | std: {}'.format(d, means[d], variances[d], stds[d]))
        diff = variances[d] - np.var(results[d])
        assert (diff < 0.0001)

def test_on_model():
    path_model_dir = 'saved_models/dna'
    
    model = fnet.load_model_from_dir(path_model_dir)
    print(model)
    dataset = fnet.data.load_dataset_from_dir(path_model_dir)
    dataset.use_test_set()
    print(dataset)

    n_distances = 16
    # dims_chunk = (32, n_distances*2, n_distances*2)
    dims_chunk = (32, 32, 32)
    
    data_provider = fnet.data.ChunkDataProvider(
        dataset,
        dims_chunk = dims_chunk,
        buffer_size = 1,
        batch_size = 24,
        replace_interval=-1,
    )

    keys = [0, 1, 2, (0, 1, 2)]

    map_sums_target = {}
    map_sums_of_squares_target = {}
    map_counts_target = {}
    
    map_sums_pred = {}
    map_sums_of_squares_pred = {}
    map_counts_pred = {}
    
    map_sums_se = {}
    map_sums_of_squares_se = {}
    map_counts_se = {}
    for key in keys:
        map_sums_target[key] = np.zeros(n_distances)
        map_sums_of_squares_target[key] = np.zeros(n_distances)
        map_counts_target[key] = np.zeros(n_distances)

        map_sums_pred[key] = np.zeros(n_distances)
        map_sums_of_squares_pred[key] = np.zeros(n_distances)
        map_counts_pred[key] = np.zeros(n_distances)
        
        map_sums_se[key] = np.zeros(n_distances)
        map_sums_of_squares_se[key] = np.zeros(n_distances)
        map_counts_se[key] = np.zeros(n_distances)
        
    for i in range(100):
        print('doing batch', i)
        batch_signal, batch_target = data_provider.get_batch()
        batch_prediction = model.predict(batch_signal)

        # error injection
        # batch_prediction[:, :, :, :, (3, 2*n_distances - 1 - 3)] = 0
        batch_se = (batch_prediction - batch_target)**2
        
        for key in keys:
            update_stats_arrays(batch_target[:, 0, ...], key, map_sums_target[key], map_sums_of_squares_target[key], map_counts_target[key])
            update_stats_arrays(batch_prediction[:, 0, ...], key, map_sums_pred[key], map_sums_of_squares_pred[key], map_counts_pred[key])
            update_stats_arrays(batch_se[:, 0, ...], key, map_sums_se[key], map_sums_of_squares_se[key], map_counts_se[key])
            
    means_target, stds_target = [], []
    means_pred, stds_pred = [], []
    means_se, stds_se = [], []

    norm_squared_errors = []
    legend = list(map_sums_target.keys())
    for key in legend:
        ar_mean_target, ar_std_target = get_means_stds(map_sums_target[key], map_sums_of_squares_target[key], map_counts_target[key])
        means_target.append(ar_mean_target)
        stds_target.append(ar_std_target)

        ar_mean_pred, ar_std_pred = get_means_stds(map_sums_pred[key], map_sums_of_squares_pred[key], map_counts_pred[key])
        means_pred.append(ar_mean_pred)
        stds_pred.append(ar_std_pred)
        
        ar_mean_se, ar_std_se = get_means_stds(map_sums_se[key], map_sums_of_squares_se[key], map_counts_se[key])
        means_se.append(ar_mean_se)
        stds_se.append(ar_std_se)

        norm_squared_errors.append(map_sums_se[key]/map_sums_of_squares_target[key])
        
    path_save_target = 'tmp/tests/outputs/target.png'
    make_plot(path_save_target, means_target, stds_target, ylabel='mean intensity target', legend=legend)
    
    path_save_pred = 'tmp/tests/outputs/prediction.png'
    make_plot(path_save_pred, means_pred, stds_pred, ylabel='mean intensity prediction', legend=legend)

    path_save = 'tmp/tests/outputs/mse.png'
    make_plot(path_save, means_se, stds_se, ylabel='mean squared error', legend=legend)
    
    path_save_se_norm = 'tmp/tests/outputs/normalized_se.png'
    make_plot(path_save_se_norm, norm_squared_errors, None, ylabel='normalized squared error', legend=legend)

if __name__ == '__main__':
    # test_get_mask()
    # test_plot()
    test_on_model()
    
