import os
from pathlib import Path
from fft_utils import complex_2_numpy
import numpy as np
import imageio
import torch

def zero_mean_norm_ball(x, zero_mean=True, normalize=True, norm_bound=1.0, norm='l2', mask=None, axis=(0,...)):
    """ project onto zero-mean and norm-one ball
    :param x: tf variable which should be projected
    :param zero_mean: boolean True for zero-mean. default=True
    :param normalize: boolean True for l_2-norm ball projection. default:True
    :param norm_bound: defines the size of the norm ball
    :param norm: type of the norm
    :param mask: binary mask to compute the mean and norm
    :param axis: defines the axis for the reduction (mean and norm)
    :return: projection ops
    
    - modified: x (convolutional kernel) is assumed to be 2-channel complex tensor
    from https://github.com/VLOGroup/mri-variationalnetwork/blob/master/vn/proxmaps.py
    """
    if mask is None:
        shape = []
        for i in range(len(x.shape)):
            if i in axis:
                shape.append(x.shape[i])
            else:
                shape.append(1)
        mask = torch.ones(shape, dtype=torch.float32)

    
    x_masked = x * mask

    if zero_mean:
        x_mean = torch.mean(x_masked,dim=axis,keepdim=True)
        x_zm = x_masked - x_mean
    else:
        x_zm = x_masked

    if normalize:
        if norm == 'l2':
            magnitude = torch.sqrt(torch.sum(torch.square(x_zm),dim=axis,keepdim=True))
            x_proj = x_zm / magnitude * norm_bound   
        else:
            raise ValueError("Norm '%s' not defined." % norm)
            x_proj = x_zm

    return x_proj

def print_options(parser, opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    taken from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True,exist_ok=True)
    file_name =  save_dir / '{}_opt.txt'.format(opt.mode)
    with open(str(file_name), 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def save_recon(under, recon, reference, index, save_dir, error_scale=1,do_save=True):
    """ Save the reconstruction result and compare with reference

    Parameters:
    ----------
    under: torch tensor (NxHxWx2)
        complex undersampled image
    recon: torch tensor (NxHxWx2)
        complex reconstruction from network
    reference: torch tensor (NxHxWx2)
        complex reference fully sampled image
    index: int
        index of the current batch
    save_dir: pathlib.Path
        location to save the image
    do_save: bool
        to write the image to save_dir or not (use during inference)
    error_scale: float
        how much to magnify the error map

    Outputs:
    ----------
    None. The magnitude image of the slice are saved at save_dir
    """
    batch_size = under.shape[0]
    under_np = complex_2_numpy(under.detach().cpu())
    recon_np = complex_2_numpy(recon.detach().cpu())
    reference_np = complex_2_numpy(reference.detach().cpu())

    for i in range(batch_size):
        under_mag = np.abs(under_np[i])
        recon_mag = np.abs(recon_np[i])
        reference_mag = np.abs(reference_np[i])

        diff = error_scale*np.abs(reference_mag - recon_mag)

        img_to_save = 255*np.concatenate((under_mag,recon_mag,reference_mag,diff),axis=1)
        if do_save:
            save_name = save_dir / ('{}.png'.format(batch_size*index + i))
            imageio.imwrite(str(save_name),img_to_save.astype(np.uint8)) 

    return img_to_save


