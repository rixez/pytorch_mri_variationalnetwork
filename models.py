import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from fft_utils import *
from misc_utils import *
import copy
from pathlib import Path
from optimizer import IIPG

DEFAULT_OPTS = {'kernel_size':11,
                'features_in':1,
                'features_out':24,
                'do_prox_map':True,
                'pad':11,
                'vmin':-1.0,'vmax':1.0,
                'lamb_init':1.0,
                'num_act_weights':31,
                'init_type':'linear',
                'init_scale':0.04,
                'sampling_pattern':'cartesian',
                'num_stages':10,
                'seed':1,
                'optimizer':'adam','lr':1e-4,
                'activation':'rbf',
                'loss_type':'complex', 
                'momentum':0.,
                'error_scale':10,
                'loss_weight':1}



class RBFActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w, mu, sigma):
        """ Forward pass for RBF activation

        Parameters:
        ----------
        ctx: 
        input: torch tensor (NxCxHxW)
            input tensor
        w: torch tensor (1 x C x 1 x 1 x # of RBF kernels)
            weight of the RBF kernels
        mu: torch tensor (# of RBF kernels)
            center of the RBF
        sigma: torch tensor (1)
            std of the RBF

        Returns:
        ----------
        torch tensor: linear weight combination of RBF of input
        """
        num_act_weights = w.shape[-1]
        output = input.new_zeros(input.shape)
        rbf_grad_input = input.new_zeros(input.shape)
        for i in range(num_act_weights):
            tmp = w[:,:,:,:,i] * torch.exp(-torch.square(input - mu[i])/(2* sigma ** 2))
            output += tmp
            rbf_grad_input += tmp*(-(input-mu[i]))/(sigma**2)
        del tmp
        ctx.save_for_backward(input,w,mu,sigma,rbf_grad_input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w, mu, sigma, rbf_grad_input = ctx.saved_tensors
        num_act_weights = w.shape[-1]

        #if ctx.needs_input_grad[0]:
        grad_input = grad_output * rbf_grad_input

        #if ctx.need_input_grad[1]:
        grad_w = w.new_zeros(w.shape)
        for i in range(num_act_weights):
            tmp = (grad_output*torch.exp(-torch.square(input-mu[i])/(2*sigma**2))).sum((0,2,3))
            grad_w[:,:,:,:,i] = tmp.view(w.shape[0:-1])
    
        return grad_input, grad_w, None, None


class RBFActivation(nn.Module):
    """ RBF activation function with trainable weights """
    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs
        x_0 = np.linspace(kwargs['vmin'],kwargs['vmax'],kwargs['num_act_weights'],dtype=np.float32)
        mu = np.linspace(kwargs['vmin'],kwargs['vmax'],kwargs['num_act_weights'],dtype=np.float32)
        self.sigma = 2*kwargs['vmax']/(kwargs['num_act_weights'] - 1)
        self.sigma = torch.tensor(self.sigma)
        if kwargs['init_type'] == 'linear':
            w_0 = kwargs['init_scale']*x_0
        elif kwargs['init_type'] == 'tv':
            w_0 = kwargs['init_scale'] * np.sign(x_0)
        elif kwargs['init_type'] == 'relu':
            w_0 = kwargs['init_scale'] * np.maximum(x_0, 0)
        elif kwargs['init_type'] == 'student-t':
            alpha = 100
            w_0 = kwargs['init_scale'] * np.sqrt(alpha)*x_0/(1+0.5*alpha*x_0**2)
        else:
            raise ValueError("init_type '%s' not defined!" % kwargs['init_type'])
        w_0 = np.reshape(w_0,(1,1,1,1,kwargs['num_act_weights']))
        w_0 = np.repeat(w_0,kwargs['features_out'],1)
        self.w = torch.nn.Parameter(torch.from_numpy(w_0))
        self.mu = torch.from_numpy(mu)
        self.rbf_act = RBFActivationFunction.apply

    def forward(self,x):
        # x = x.unsqueeze(-1)
        # x = x.repeat((1,1,1,1,self.mu.shape[-1]))
        # if not self.mu.device == x.device:
        #     self.mu = self.mu.to(x.device)
        #     self.std = self.std.to(x.device)
        # gaussian = torch.exp(-torch.square(x - self.mu)/(2*self.std ** 2))
        # weighted_gaussian = self.w_0 * gaussian
        # out = torch.sum(weighted_gaussian,axis=-1,keepdim=False)
        if not self.mu.device == x.device:
        	self.mu = self.mu.to(x.device)
        	self.sigma = self.sigma.to(x.device)

        # out = torch.zeros(x.shape,dtype=torch.float32,device=x.device)
        # for i in range(self.options['num_act_weights']):
        # 	out += self.w_0[:,:,:,:,i] * torch.exp(-torch.square(x - self.mu[:,:,:,:,i])/(2*self.std ** 2))
        output = self.rbf_act(x,self.w,self.mu,self.sigma)
        	
        return output

class VnMriReconCell(nn.Module):
    """ One cell of variational network """
    def __init__(self, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options
        conv_kernel = np.random.randn(options['features_out'],options['features_in'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*options['features_in'])
        conv_kernel -= np.mean(conv_kernel, axis=(1,2,3,4),keepdims=True)
        conv_kernel = torch.from_numpy(conv_kernel)
        if options['do_prox_map']:
            conv_kernel = zero_mean_norm_ball(conv_kernel,axis=(1,2,3,4))


        self.conv_kernel = torch.nn.Parameter(conv_kernel)

        if self.options['activation'] == 'rbf':
            self.activation = RBFActivation(**options)
        elif self.options['activation'] == 'relu':
            self.activation = torch.nn.ReLU()
        self.lamb = torch.nn.Parameter(torch.tensor(options['lamb_init'],dtype=torch.float32))


    def mri_forward_op(self, u, coil_sens, sampling_mask, os=False):
        """
        Forward pass with kspace
        (2X the size)
        
        Parameters:
        ----------
        u: torch tensor NxHxWx2
            complex input image
        coil_sens: torch tensor NxCxHxWx2
            coil sensitivity map
        sampling_mask: torch tensor NxHxW
            sampling mask to undersample kspace
        os: bool
            whether the data is oversampled in frequency encoding

        Returns:
        -----------
        kspace of u with applied coil sensitivity and sampling mask
        """
        if os:
            pad_u = torch.tensor((sampling_mask.shape[1]*0.25 + 1),dtype=torch.int16)
            pad_l = torch.tensor((sampling_mask.shape[1]*0.25 - 1),dtype=torch.int16)
            u_pad = F.pad(u,[0,0,0,0,pad_u,pad_l])
        else:
            u_pad = u
        u_pad = u_pad.unsqueeze(1)
        coil_imgs = complex_mul(u_pad, coil_sens) # NxCxHxWx2
        
        Fu = fftc2d(coil_imgs) #
        
        mask = sampling_mask.unsqueeze(1) # Nx1xHxW
        mask = mask.unsqueeze(4) # Nx1xHxWx1
        mask = mask.repeat([1,1,1,1,2]) # Nx1xHxWx2

        kspace = mask*Fu # NxCxHxWx2
        return kspace

    def mri_adjoint_op(self, f, coil_sens, sampling_mask, os=False):
        """
        Adjoint operation that convert kspace to coil-combined under-sampled image
        by using coil_sens and sampling mask
        
        Parameters:
        ----------
        f: torch tensor NxCxHxWx2
            multi channel kspace
        coil_sens: torch tensor NxCxHxWx2
            coil sensitivity map
        sampling_mask: torch tensor NxHxW
            sampling mask to undersample kspace
        os: bool
            whether the data is oversampled in frequency encoding
        Returns:
        -----------
        Undersampled, coil-combined image
        """
        
        # Apply mask and perform inverse centered Fourier transform
        mask = sampling_mask.unsqueeze(1) # Nx1xHxW
        mask = mask.unsqueeze(4) # Nx1xHxWx1
        mask = mask.repeat([1,1,1,1,2]) # Nx1xHxWx2

        Finv = ifftc2d(mask*f) # NxCxHxWx2
        # multiply coil images with sensitivities and sum up over channels
        img = torch.sum(complex_mul(Finv,conj(coil_sens)),1)

        if os:
            # Padding to remove FE oversampling
            pad_u = torch.tensor((sampling_mask.shape[1]*0.25 + 1),dtype=torch.int16)
            pad_l = torch.tensor((sampling_mask.shape[1]*0.25 - 1),dtype=torch.int16)
            img = img[:,pad_u:-pad_l,:,:]
            
        return img

    def forward(self, inputs):
        u_t_1 = inputs['u_t'] #NxHxWx2
        f = inputs['f']
        c = inputs['coil_sens']
        m = inputs['sampling_mask']

        u_t_1 = u_t_1.unsqueeze(1) #Nx1xHxWx2
        # pad the image to avoid problems at the border
        pad = self.options['pad']
        u_t_real = u_t_1[:,:,:,:,0]
        u_t_imag = u_t_1[:,:,:,:,1]
        
        u_t_real = F.pad(u_t_real,[pad,pad,pad,pad],mode='reflect') #to do: implement symmetric padding
        u_t_imag = F.pad(u_t_imag,[pad,pad,pad,pad],mode='reflect')
        # split the image in real and imaginary part and perform convolution
        u_k_real = F.conv2d(u_t_real,self.conv_kernel[:,:,:,:,0],stride=1,padding=5)
        u_k_imag = F.conv2d(u_t_imag,self.conv_kernel[:,:,:,:,1],stride=1,padding=5)
        # add up the convolution results
        u_k = u_k_real + u_k_imag
        #apply activation function
        f_u_k = self.activation(u_k)
        # perform transpose convolution for real and imaginary part
        u_k_T_real = F.conv_transpose2d(f_u_k,self.conv_kernel[:,:,:,:,0],stride=1,padding=5)
        u_k_T_imag = F.conv_transpose2d(f_u_k,self.conv_kernel[:,:,:,:,1],stride=1,padding=5)

        #Rebuild complex image
        u_k_T_real = u_k_T_real.unsqueeze(-1)
        u_k_T_imag = u_k_T_imag.unsqueeze(-1)
        u_k_T =  torch.cat((u_k_T_real,u_k_T_imag),dim=-1)

        #Remove padding and normalize by number of filter
        Ru = u_k_T[:,0,pad:-pad,pad:-pad,:] #NxHxWx2
        Ru /= self.options['features_out']

        if self.options['sampling_pattern'] == 'cartesian':
            os = False
        elif not 'sampling_pattern' in self.options or self.options['sampling_pattern'] == 'cartesian_with_os':
            os = True

        Au = self.mri_forward_op(u_t_1[:,0,:,:,:],c,m,os)
        At_Au_f = self.mri_adjoint_op(Au - f, c, m,os)
        Du = At_Au_f * self.lamb
        u_t = u_t_1[:,0,:,:,:] - Ru - Du
        output = {'u_t':u_t,'f':f,'coil_sens':c,'sampling_mask':m}
        return output #NxHxWx2

class VariationalNetwork(pl.LightningModule):   
    def __init__(self,**kwargs):
        super().__init__()
        options = DEFAULT_OPTS

        for key in kwargs.keys():
            options[key] = kwargs[key]

        self.options = options
        cell_list = []
        for i in range(options['num_stages']):
            cell_list.append(VnMriReconCell(**options))

        self.cell_list = nn.Sequential(*cell_list)
        self.log_img_count = 0


    def forward(self,inputs):
        output = self.cell_list(inputs)
        return output['u_t']
    
    def training_step(self, batch, batch_idx):
        recon_img = self(batch)
        ref_img = batch['reference']
        
        if self.options['loss_type'] == 'complex':
            loss = F.mse_loss(recon_img,ref_img)
        elif self.options['loss_type'] == 'magnitude':
            recon_img_mag = torch_abs(recon_img)
            ref_img_mag = torch_abs(ref_img)    
            loss = F.mse_loss(recon_img_mag,ref_img_mag)
        loss = self.options['loss_weight']*loss
        if batch_idx % (int(200/self.options['batch_size']/4)) == 0:
            sample_img = save_recon(batch['u_t'],recon_img,ref_img,batch_idx,'save_dir',self.options['error_scale'],False)
            sample_img = sample_img[np.newaxis,:,:]
            self.logger.experiment.add_image('sample_recon',sample_img.astype(np.uint8),self.log_img_count)
            self.log_img_count += 1

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        recon_img = self(batch)
        ref_img = batch['reference']
        recon_img_mag = torch_abs(recon_img)
        ref_img_mag = torch_abs(ref_img)
        loss = F.mse_loss(recon_img_mag,ref_img_mag)
        img_save_dir = Path(self.options['save_dir']) / ('eval_result_img_' + self.options['name'])
        img_save_dir.mkdir(parents=True,exist_ok=True)
        save_recon(batch['u_t'],recon_img,ref_img,batch_idx,img_save_dir,self.options['error_scale'],True)
        return {'test_loss':loss}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}
    

    def configure_optimizers(self):
        if self.options['optimizer'] == 'adam':
            return torch.optim.Adam(self.parameters(),lr=self.options['lr'])
        elif self.options['optimizer'] == 'sgd':
        	return torch.optim.SGD(self.parameters(),lr=self.options['lr'],momentum=self.options['momentum'])
        elif self.options['optimizer'] == 'rmsprop':
        	return torch.optim.RMSprop(self.parameters(),lr=self.options['lr'],momentum=self.options['momentum'])
        elif self.options['optimizer'] == 'iipg':
            iipg = IIPG(torch.optim.SGD,self.parameters(),lr=self.options['lr'],momentum=self.options['momentum'])
            return iipg

