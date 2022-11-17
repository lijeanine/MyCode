import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import nibabel as nib

class LoadMRIData(Dataset):
    
    def __init__(self, mri_dir, list_dir, phase, num_class, num_slices=5, se_loss = True, Encode3D = False, use_weight = False):
        "load MRI into a 2D slice and a 3D image"
        
        self.phase = phase
        self.se_loss = se_loss
        self.Encode3D = Encode3D
        self.num_class = num_class
        self.use_weight = use_weight
        self.num_slices = num_slices
        
        if self.use_weight:
            weight_dir = os.path.join(mri_dir, 'training-weightsnpy')
            self.weight_names = []
        
        if self.phase is 'train':
            data_dir = os.path.join(mri_dir, 'training-imagesnpy')
            if num_class is 10:
                label_dir = os.path.join(mri_dir, 'training-labels-remapnpy9')
            else:
                label_dir = os.path.join(mri_dir, 'training-labels139')
            image_list = os.path.join(list_dir, 'train_volumes.txt')
            
            self.image_names = []
            self.image_slices = []
            self.label_names = []
            #self.skull_names = []
            with open(image_list, 'r') as f:
                for line in f:
                    for i in range(256):
                        image_name = os.path.join(data_dir, line.rstrip() + '.npy')
                        label_name = os.path.join(label_dir, line.rstrip() + '_glm.npy')
                        #skull_name = os.path.join(data_dir, line.rstrip() + '_brainmask.npy')
                        self.image_names.append(image_name)
                        self.label_names.append(label_name)
                        #self.skull_names.append(skull_name)
                        self.image_slices.append(i)
                        
                        if self.use_weight:
                            weight_name = os.path.join(weight_dir, line.rstrip() + '_glm.npy')
                            self.weight_names.append(weight_name)
        elif self.phase is 'test':
            data_dir = os.path.join(mri_dir, 'testing-imagesnpy')
            if num_class is 10:
                label_dir = os.path.join(mri_dir, 'testing-labels-remapnpy9')
            else:
                label_dir = os.path.join(mri_dir, 'testing-labels139')
            image_list = os.path.join(list_dir, 'test_volumes.txt')
            
            self.image_names = []
            self.image_slices = []
            self.label_names = []
            with open(image_list, 'r') as f:
                for line in f:
                    image_name = os.path.join(data_dir, line.rstrip() + '.npy')
                    label_name = os.path.join(label_dir, line.rstrip() + '_glm.npy')
                    self.image_names.append(image_name)
                    self.label_names.append(label_name)

    
    def __getitem__(self, idx):
        #this is for non-pre-processing data        
        image_name = self.image_names[idx]
        label_name = self.label_names[idx]
        
        img_3D = np.load(image_name)
        #normalize data
        img_3D = (img_3D.astype(np.float32)-128)/128
        label_3D = np.load(label_name).astype(np.int32)
        
        if self.phase is 'train':
            x_ax, y_ax, z_ax = np.shape(img_3D)
            #skull_name = self.skull_names[idx]
            #skull_3D = np.load(skull_name).astype(np.int32)
            
            image_slice = self.image_slices[idx]
            img_coronal = img_3D[:,:,image_slice][np.newaxis,:,:]#[1,256,256]
            label_coronal = label_3D[:,:,image_slice]
            #skull_coronal = skull_3D[:,:,image_slice]
            
            sample = {'image': torch.from_numpy(img_coronal), 'label': torch.from_numpy(label_coronal)}
                      #'skull': torch.from_numpy(skull_coronal)}
        
            if self.se_loss:
                curlabel = np.unique(label_coronal)
                cls_logits = np.zeros(self.num_class, dtype = np.float32)
                if np.sum(curlabel > self.num_class) >0:
                    curlabel[curlabel>self.num_class] = 0
                cls_logits[curlabel] = 1
                sample['se_gt'] = torch.from_numpy(cls_logits)
            
            if self.Encode3D:
                if image_slice<=int(self.num_slices*2+1):
                    image_stack = img_3D[:,:,0:int(self.num_slices*2+1)]
                elif image_slice >=z_ax-int(self.num_slices*2+1):
                    image_stack = img_3D[:,:,z_ax-int(self.num_slices*2+1):]
                else:
                    image_stack = img_3D[:,:,image_slice-self.num_slices:image_slice+self.num_slices+1]
                image_stack = np.transpose(image_stack, (2,0,1))
                sample['image_stack'] = torch.from_numpy(image_stack)
                
            #estimate class weights
            if self.use_weight:
                weight_name = self.weight_names[idx]
                weights_3D = np.load(weight_name).astype(np.float32)
                weight_slice = weights_3D[:,:,image_slice]
                sample['weights'] = torch.from_numpy(weight_slice)
            
        if self.phase is 'test':
            img_3D = np.transpose(img_3D, (2,0,1))
            label_3D = np.transpose(label_3D, (2,0,1))
            name = image_name.split('/')[-1][:-4]
            sample = {'image_3D': torch.from_numpy(img_3D), 'label_3D': torch.from_numpy(label_3D),
                      'name': name}
            
        return sample
        
    
    def __len__(self):
        return len(self.image_names)
        
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npy')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
if __name__=='__main__':
    NUM_CLASS = 10 
    NUM_SLICES = 1 #1
    DATA_DIR = '/opt/data/private/ACEnet_dhcp/resampled_Swin_UNet/'
    DATA_LIST = '/opt/data/private/ACEnet_dhcp/datasets/'
    db_train = LoadMRIData(DATA_DIR, DATA_LIST, 'train', NUM_CLASS, num_slices=NUM_SLICES, se_loss = False, use_weight = False, Encode3D=True)
    n = len(db_train)
    for i in range(9):
       out = db_train[i]['label'] #__getitem__ : python built-in method
       print(out.shape)
    
