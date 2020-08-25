import torch
from torch.optim import Optimizer, SGD
from misc_utils import *

DEFAULT_OPTS = {'momentum': 0.}

class IIPG(Optimizer):
	""" Inertial Incremental proximal gradient optimizer as described in Hammernik et al 2016
	Simple implementation that use an existing optimizer and perform proximal mapping after gradient descent
	"""
	def __init__(self, optimizer, params, **kwargs):
		self.options = DEFAULT_OPTS
		for key in kwargs:
			self.options[key] = kwargs[key]
		self.opt = optimizer(params, **kwargs)
		self.params = params
		self.param_groups = self.opt.param_groups
		self.state = self.opt.state
	@torch.no_grad()
	def step(self, closure=None):
		loss = self.opt.step(closure)
		# projected gradient descent
		for param in self.params:	
			if len(param.shape) == 0:
				# data weight term
				param.data = torch.max(input = param.data, other = torch.zeros_like(param.data))

			else:
				if param.shape[0] > 1:
					# convolutional kernel
					param.data = zero_mean_norm_ball(param_data, axis=(1,2,3,4))

		return loss


