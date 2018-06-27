import numpy as np
import torch
import pdb

import itertools

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.stats import multivariate_normal as mvn

def get_slice(starts, ends, n_channels = 1):
    index = [slice(s, e) for s,e in zip(starts,ends)]
    index = [slice(0,1), slice(0, n_channels)] + index
    
    patch_index = tuple(index)
    
    return patch_index

def get_triangle_kernel(patch_size, n_channels):
    
    starts = np.ones(patch_size.size).astype(int)
    ends = patch_size-1
    
    kernel = np.zeros(patch_size)
    
    kernel[get_slice(starts, ends, n_channels)[2:]] = 1
    kernel = edt(kernel) + 1
    
    kernel = np.expand_dims(np.expand_dims(kernel, 0),0)
    kernel = np.repeat(kernel, n_channels, 1)
    
    kernel = torch.Tensor(kernel)
    
    return kernel

def get_gaussian_kernel(patch_size, n_channels, n_stdev = 2):
    
    grid = np.meshgrid(*[np.linspace(-n_stdev, n_stdev, dim) for dim in patch_size])
    grid = [dim.flatten() for dim in grid]
    grid = np.vstack(grid).T
    
    grid_pdfs = mvn.pdf(grid, np.zeros(len(patch_size)), np.ones(len(patch_size)))
    
    kernel = grid_pdfs.reshape(patch_size)
    
    kernel = np.expand_dims(np.expand_dims(kernel, 0),0)
    kernel = np.repeat(kernel, n_channels, 1)
    
    kernel = torch.Tensor(kernel)
    
    return kernel

def predict_stitched(model, image, patch_size = [64, 64, 64], step_size = [32, 32, 32], kernel_type = 'gaussian'):
    
    patch_size = np.array(patch_size)
    

    
    step_size = np.array(step_size)
    
    patch_subsample = ((patch_size - step_size) /2).astype(int)
    
    imsize = np.array([*image.size()][2:])
    patch_size[patch_size>imsize] = imsize[patch_size>imsize]
        
    steps = list()
    for dim, dim_step, patch in zip(imsize, step_size, patch_size):
        #make sure we get a _full_ patch for the last sample
        
        starts = np.hstack([np.arange(0, dim-patch, dim_step), dim-patch])
        steps += [starts]

    #this is probably the most straightforward way of getting the number of output channels
    starts = np.zeros(len(imsize)).astype(int)
    ends = starts + patch_size
    index = get_slice(starts, ends)
    n_channels = model.predict(image[index]).shape[1]
    
    imsize = np.array([*image.size()])
    imsize[1] = n_channels
 
    if kernel_type is None:
        kernel = None
    elif kernel_type is 'triangle':
        kernel = get_triangle_kernel(patch_size, n_channels).type_as(image)
    elif kernel_type is 'gaussian':
        kernel = get_gaussian_kernel(patch_size, n_channels).type_as(image)
    else:
        raise ValueError("kernel_type must be None, 'triangle', or 'gaussian'")

    image_out = torch.zeros(tuple(imsize)).type_as(image)
    patch_counts = torch.zeros(tuple(imsize)).type_as(image)
        
    #loop all permutations of our gridded sampling
    for step in itertools.product(*steps):
    
        #get the patch
        source_starts = np.array(step)
        source_ends = source_starts + patch_size
        source_slice = get_slice(source_starts, source_ends, n_channels)
        image_patch = image[source_slice]
    
        #predict
        with torch.no_grad(): 
            out_patch = model.predict(image_patch)
        
        #figure out where we need to crop the output patch
        is_start = source_starts == 0
        is_end = source_ends == imsize[2:]

        if kernel is None:
            patch_starts = patch_subsample.copy()
            patch_ends = patch_starts + step_size
            
            patch_starts[is_start] = 0
            patch_ends[is_end] = patch_size[is_end]
            
            patch_slice = get_slice(patch_starts, patch_ends, n_channels)
            out_patch = out_patch[patch_slice]

        #figure out where we need to place the output patch in the output image
        
        if kernel is None:
            out_starts = np.array(step)+patch_subsample
            out_ends = out_starts + step_size
            
            out_starts[is_start] = 0
            out_ends[is_end] = imsize[2:][is_end]
        else:
            out_starts = step
            out_ends = step+patch_size    
            
        output_slice = get_slice(out_starts, out_ends, n_channels)

        if kernel is None:
            image_out[output_slice] += out_patch
            patch_counts[output_slice] += 1
        else:
            try:
                image_out[output_slice] += out_patch*kernel
            except:
                pdb.set_trace()
            patch_counts[output_slice] += kernel
        

    image_out = image_out/patch_counts
    
    return image_out
    
    
    
    
    