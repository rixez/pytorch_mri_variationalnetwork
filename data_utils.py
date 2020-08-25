import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.io import loadmat
import copy
import mri_utils
from fft_utils import numpy_2_complex
import numpy as np
DEFAULT_OPTS = {'root_dir':'data/knee',
				'name':'coronal_pd', 
				'patients':[1,2,3,4,5,6,7,8,9,10],
				'start_slice':11,'end_slice':30,
				'eval_patients':[11,12,13,14,15,16,17,18,19,20],
				'eval_slices':[x for x in range(10,30)],'mode':'train',
				'load_target':True,'sampling_pattern':'cartesian_with_os',
				'normalization':'max'} 

class KneeDataset(Dataset):
	""" MRI knee dataset with k-space raw data, coil sensitivities and sampling mask
	Adapted from Hammernik et al """
	def __init__(self, **kwargs):
		"""
		Parameters:
		root_dir: str
			root directory of data
		dataset_name: list of str
			list of directory to load data from
		transform: 
		"""
		options = DEFAULT_OPTS

		for key in kwargs.keys():
			options[key] = kwargs[key]

		self.options = options
		self.root_dir = Path(self.options['root_dir'])
		# Processing directory
		if not options['name'] in ['coronal_pd','axial_t2','coronal_pd_fs','sagittal_pd','sagittal_t2']:
			raise ValueError('Dataset {} not supported!'.format(options['name']))

		self.filename = []
		self.coil_sens_list = []
		data_dir = self.root_dir / options['name']

		# Load raw data and coil sensitivities name
		if options['mode'] == 'train':
			patient_key = 'patients'
			slice_no = [x for x in range(options['start_slice'],options['end_slice']+1)]
		elif options['mode'] == 'eval':
			patient_key = 'eval_patients'
			slice_no = options['eval_slices']

		for patient in options[patient_key]:
			patient_dir = data_dir / str(patient)
			for i in slice_no:
				slice_dir = patient_dir / 'rawdata{}.mat'.format(i)
				self.filename.append(str(slice_dir))
				coil_sens_dir = patient_dir / 'espirit{}.mat'.format(i)
				self.coil_sens_list.append(str(coil_sens_dir))

		
		self.mask_dir = data_dir/ 'masks'
		self.mask_dir = list(self.mask_dir.glob('*at4*'))
		self.mask = loadmat(str(self.mask_dir[0]))
		self.mask = self.mask['mask'].astype(np.float32)

	def __len__(self):
		return len(self.filename)

	def __getitem__(self,idx):
		mask = copy.deepcopy(self.mask)
		filename = self.filename[idx]
		coil_sens = self.coil_sens_list[idx]

		raw_data = loadmat(filename)
		f = np.ascontiguousarray(np.transpose(raw_data['rawdata'],(2,0,1))).astype(np.complex64)
		
		coil_sens_data = loadmat(coil_sens)
		c = np.ascontiguousarray(np.transpose(coil_sens_data['sensitivities'],(2,0,1))).astype(np.complex64)

		if self.options['load_target']:
			ref = coil_sens_data['reference'].astype(np.complex64)
		else:
			ref = np.zeros_like(mask,dtype=np.complex64)

		if 'padlength_left' in raw_data and 'padlength_right' in raw_data:
			padlength_left = int(raw_data['padlength_left'])
			padlength_right = int(raw_data['padlength_right'])
		else:
			padlength_left = 0
			padlength_right = 0

		if padlength_left > 0:
			mask[:,:padlength_left] = 1
		if padlength_right  > 0:
			mask[:,-padlength_right:] = 1

		# mask rawdata
		f *= mask

		# compute initial image input0
		input0 = mri_utils.mriAdjointOp(f,c,mask).astype(np.complex64)

		# remove frequency encoding oversampling
		if self.options['sampling_pattern'] == 'cartesian_with_os':
			if self.options['load_target']:
				ref = mri_utils.removeFEOversampling(ref) # remove RO Oversampling
			input0 = mri_utils.removeFEOversampling(input0) # remove RO Oversampling

		elif self.options['sampling_pattern'] == 'cartesian':
			pass
		else:
			raise ValueError('sampling_pattern has to be in [cartesian_with_os, cartesian]')

		# normalize the data
		if self.options['normalization'] == 'max':
			norm = np.max(np.abs(input0))
		elif self.options['normalization'] == 'no':
			norm = 1.0
		else:
			raise ValueError("Normalization has to be in ['max','no']")

		f /= norm
		input0 /= norm

		if self.options['load_target']:
			ref /= norm
		else:
			ref = np.zeros_like(input_0)

		input0 = numpy_2_complex(input0)
		f = numpy_2_complex(f)
		c = numpy_2_complex(c)
		mask = torch.from_numpy(mask)
		ref = numpy_2_complex(ref)

		data = {'u_t':input0,'f':f,'coil_sens':c,'sampling_mask':mask,'reference':ref}
		return data















