import numpy as np
import nibabel as nib
img_path = 'xxx/xxx.npy'
img = np.load(img_path)
new_image = nib.Nifti1Image(img, np.eye(4))
nib.save(new_image, 'xxx.nii.gz')
