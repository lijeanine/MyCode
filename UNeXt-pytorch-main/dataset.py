import os

import cv2
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset

class LoadMRIData(torch.utils.data.Dataset):
    '''
    code copying from Swin-UNet
    '''
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
                    for i in range(150):#ori:256 for resampled_dhcp
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
            data_dir = os.path.join(mri_dir, 'testing-imagesnpy') #resampled_Swin_UNet:'testing-imagesnpy',resampled_dhcp:'testing-imagesT2npy'
            if num_class is 10:
                label_dir = os.path.join(mri_dir, 'testing-labels-remapnpy9')
            else:
                label_dir = os.path.join(mri_dir, 'testing-labels139')
            image_list = os.path.join(list_dir, 'test_volumes.txt')
            
            self.image_names = []#[img1,img1,img1,...,img2,img2,...,img3,img3,...]
            self.image_slices = []#[0,1,2,...,0,1,...,0,1,...]
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
            img_coronal = img_3D[:,:,image_slice][np.newaxis,:,:]#【1,256,256】
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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}
