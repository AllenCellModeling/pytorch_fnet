import numpy as np
import matplotlib.pyplot as plt
from fnet import find_z_of_max_slice
import os
import pdb

def show_img(ar):
    import PIL
    import PIL.ImageOps
    from IPython.core.display import display
    img_norm = ar - ar.min()
    img_norm *= 255./img_norm.max()
    img_pil = PIL.Image.fromarray(img_norm).convert('L')
    display(img_pil)
    
def draw_rect(img, coord_tl, dims_rect, thickness=3, color=0):
    """Draw rectangle on image.

    Parameters:
    img - 2d numpy array (image is modified)
    coord_tl - coordinate within img to be top-left corner or rectangle
    dims_rect - 2-value tuple indicated the dimensions of the rectangle

    Returns:
    None
    """
    assert len(img.shape) == len(coord_tl) == len(dims_rect) == 2
    for i in range(thickness):
        if (i+1)*2 <= dims_rect[0]:
            # create horizontal lines
            img[coord_tl[0] + i, coord_tl[1]:coord_tl[1] + dims_rect[1]] = color
            img[coord_tl[0] + dims_rect[0] - 1 - i, coord_tl[1]:coord_tl[1] + dims_rect[1]] = color
        if (i+1)*2 <= dims_rect[1]:
            # create vertical lines
            img[coord_tl[0]:coord_tl[0] + dims_rect[0], coord_tl[1] + i] = color
            img[coord_tl[0]:coord_tl[0] + dims_rect[0], coord_tl[1] + dims_rect[1] - 1 - i] = color

def display_batch(vol_light_np, vol_nuc_np, batch):
    """Display images of examples from batch.
    vol_light_np - numpy array
    vol_nuc_np - numpy array
    batch - 3-element tuple: chunks from vol_light_np, chunks from vol_nuc_np, coordinates of chunks
    """
    n_chunks = batch[0].shape[0]
    z_max_big = find_z_of_max_slice(vol_nuc_np)
    img_light, img_nuc = vol_light_np[z_max_big], vol_nuc_np[z_max_big]
    chunk_coord_list = batch[2]
    dims_rect = batch[0].shape[-2:]  # get size of chunk in along yz plane
    min_light, max_light = np.amin(vol_light_np), np.amax(vol_light_np)
    min_nuc, max_nuc = np.amin(vol_nuc_np), np.amax(vol_nuc_np)
    for i in range(len(chunk_coord_list)):
        coord = chunk_coord_list[i][1:]  # get yx coordinates
        draw_rect(img_light, coord, dims_rect, thickness=2, color=min_light)
        draw_rect(img_nuc, coord, dims_rect, thickness=2, color=min_nuc)

    print('light volume slice | z =', z_max_big)
    show_img(img_light)
    print('-----')
    print('nuc volume slice | z =', z_max_big)
    show_img(img_nuc)

    # display chunks
    z_mid_chunk = batch[0].shape[2]//2  # z-dim
    for i in range(n_chunks):
        title_str = 'chunk: ' + str(i) + ' | z:' + str(z_mid_chunk)
        fig = plt.figure(figsize=(9, 4))
        fig.suptitle(title_str)
        img_chunk_sig = batch[0][i, 0, z_mid_chunk, ]
        img_chunk_tar = batch[1][i, 0, z_mid_chunk, ]
        ax = fig.add_subplot(1, 2, 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(img_chunk_sig, cmap='gray', interpolation='bilinear')
        ax = fig.add_subplot(1, 2, 2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(img_chunk_tar, cmap='gray', interpolation='bilinear')
        plt.show()

def display_eval_images(
        sources,
        z_display=None,
        titles=('signal', 'target', 'predicted'),
        vmins=None,
        vmaxs=None,
        path_save_dir=None,
        z_save=None,
):
    """Display row of images.

    Parameters:
    sources - list/tuple of image sources, each being a 3d numpy array
    z_display - (int or iterable of ints) z-slice(s) to display
    titles - (iterable of strings) subplot titles
    vmins - (iterable) subplot vmins
    vmaxs - (iterable) subplot vmaxs
    path_save_dir - path to directory for z-stack animations
    z_save - (int or iterable of ints) z-slice(s) to save. If None, set to z_display.
    """
    scroll_bar = False
    assert isinstance(sources, (list, tuple))
    assert len(sources) == len(titles)
    if vmins is not None: assert len(vmins) == len(sources)
    if vmaxs is not None: assert len(vmaxs) == len(sources)

    z_display_list = [z_display] if isinstance(z_display, int) else z_display
    if z_save is None:
        z_render_list = z_display_list
    else:
        assert path_save_dir is not None, 'must specifiy path_save_dir if z_save is specified'
        z_render_list = [z_save] if isinstance(z_save, int) else z_save
    print('z displayed:', z_display_list)
    n_subplots = len(sources)
    if path_save_dir is not None:
        if not os.path.exists(path_save_dir):
            os.makedirs(path_save_dir)
        if scroll_bar:
            n_subplots += 1
    kwargs_list = []
    for i in range(len(sources)):
        kwargs = {}
        if vmins is not None:
            kwargs['vmin'] = vmins[i]
        if vmaxs is not None:
            kwargs['vmax'] = vmaxs[i]
        kwargs_list.append(kwargs)
    for z in z_render_list:
        fig, ax = plt.subplots(ncols=n_subplots)
        fig.set_dpi(200)
        fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0)
        for i in range(len(sources)):
            img = sources[i][z, ]
            ax[i].set_title(titles[i], fontsize='small', loc='left')
            ax[i].set_frame_on(False)
            ax[i].set_yticks([])
            ax[i].set_xticks([])
            ax[i].imshow(img, cmap='gray', interpolation='bilinear', **kwargs_list[i])
        if path_save_dir is not None:
            if scroll_bar:
                img_bar = np.ones((n_z_slices*2, n_z_slices*2), dtype=np.uint8)*255
                z_start = (n_z_slices - z - 1)*2
                img_bar[z_start: z_start + 2, :2] = 0
                ax[3].imshow(img_bar, cmap='gray')
                ax[3].axis('off')
            path_save = os.path.join(path_save_dir, 'z_{:02d}'.format(z))
            print('saving:', path_save)
            fig.savefig(path_save)
        if z in z_display_list:
            plt.show()
        plt.close(fig)
    
if __name__ == '__main__':
    print('fnet.display')
