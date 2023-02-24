#nii2npy2nii
import nibabel as nib
import os
import numpy as np
label_path = './9labels/'
label_names = os.listdir(label_path)
labelSkullStrip_path = './8labels/'
if not os.path.exists(labelSkullStrip_path):
    os.makedirs(labelSkullStrip_path)
for label_name in label_names:
    print(label_name)
    label = nib.load(label_path+label_name).get_data()
    label = np.array(label)
    for slice in range(label.shape[2]):
        for x in range(label.shape[0]):
            for y in range(label.shape[1]):
                if label[x,y,slice] == 4:
                    label[x,y,slice]=0
    new_label = nib.Nifti1Image(label,np.eye(4))
    nib.save(new_label,labelSkullStrip_path+label_name)
