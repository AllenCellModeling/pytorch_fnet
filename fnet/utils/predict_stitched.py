import numpy as np
import torch
import pdb

import itertools

def get_slice(starts, ends, n_channels = 1):
    index = [slice(s, e) for s,e in zip(starts,ends)]
    index = [slice(0,1), slice(0, n_channels)] + index
    
    patch_index = tuple(index)
    
    return patch_index

def predict_stitched(model, image, patch_size = [64, 64, 64], step_size = [32, 32, 32]):
    
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

        patch_starts = patch_subsample.copy()
        patch_ends = patch_starts + step_size

        patch_starts[is_start] = 0
        patch_ends[is_end] = patch_size[is_end]

        patch_slice = get_slice(patch_starts, patch_ends, n_channels)
        out_patch = out_patch[patch_slice]

        #figure out where we need to place the output patch in the output image
        out_starts = np.array(step)+patch_subsample
        out_ends = out_starts + step_size

        out_starts[is_start] = 0
        out_ends[is_end] = imsize[2:][is_end]
        output_slice = get_slice(out_starts, out_ends, n_channels)

        image_out[output_slice] += out_patch
        patch_counts[output_slice] += 1

    image_out = image_out/patch_counts
    
    return image_out
    
    
    
    
    