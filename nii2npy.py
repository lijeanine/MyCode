# cut size-z (256) to smaller size (e.g:192)
import nibabel as nib
import os
import numpy as np
img_path = '/opt/data/private/ACEnet_dhcp/ACEnet_dhcp_dataset/T2s/' #T2 image
label_path = '/opt/data/private/ACEnet_dhcp/ACEnet_dhcp_dataset/9labels/'

trainimg_path = './training-imagesnpy/'
trainlabel_path = './training-labels-remapnpy9/'

testimg_path = './testing-imagesnpy/'
testlabel_path = './testing-labels-remapnpy9/'
if not os.path.exists(trainimg_path):
    os.makedirs(trainimg_path)
if not os.path.exists(trainlabel_path):
    os.makedirs(trainlabel_path)
if not os.path.exists(testimg_path):
    os.makedirs(testimg_path)
if not os.path.exists(testlabel_path):
    os.makedirs(testlabel_path)
img_names = os.listdir(img_path)
for img_name in img_names:
    #1.load image and label
    print(img_name)
    label_name = str(img_name).split('.')[0] + '_glm.nii.gz'
    img = nib.load(img_path + img_name).get_data() 
    img = np.array(img)
    label = nib.load(label_path + label_name).get_data() 
    label = np.array(label)
    #2.image and label processing
    img_out = np.empty(shape=[256,256,0])
    label_out = np.empty(shape=[256,256,0])
    for slice in range(img.shape[2]): # slice on axial Z 
        if np.count_nonzero(img[:,:,slice]) and np.count_nonzero(label[:,:,slice]):
            img_out=np.append(img_out,np.expand_dims(img[:,:,slice],axis=2),axis=2)
            label_out=np.append(label_out,np.expand_dims(label[:,:,slice],axis=2),axis=2)
    #3.save processed image and label to training set and testing set 
    if str(img_name).split('.')[0] <= '25': # 3.1 save to training set 
        np.save(trainimg_path + str(img_name).split('.')[0] + '.npy', img_out) 
        np.save(trainlabel_path + str(img_name).split('.')[0] + '_glm.npy', label_out) #str(img_name).split('.')[0] = '01'
    else: #3.2 save to testing set
        np.save(testimg_path + str(img_name).split('.')[0] + '.npy', img_out) 
        np.save(testlabel_path + str(img_name).split('.')[0] + '_glm.npy', label_out)
        
        
        
        
        
        
