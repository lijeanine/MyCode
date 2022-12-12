# compute dice and iou of predicition with nii.gz format
import nibabel as nib
import os
import numpy as np
import tqdm
from collections import OrderedDict

pred_path = '.../'
label_path = '.../'
pred_names = os.listdir(pred_path)
# file for saving
tbar = tqdm(total=len(pred_names))
dice_score_list=[]
iou_score_list=[]

for pred_name in pred_names:
    #1.load pred and label
    label_name = str(pred_name).split('.')[0]+'_glm.nii.gz'
    pred = nib.load(pred_path + pred_name).get_data()
    pred = np.array(pred)
    label = nib.load(label_path + label_name).get_data()
    label = np.array(label)
    #2.compute dice and iou
    out_dice, out_iou = score_perclass(pred, label, 10)
    #3.save value of metrics
    dice_score_list.append(out_dice)
    iou_score_list.append(out_iou)
    tbar.set_description('Predict Dice Score:%.3f'%(out_dice)) #
    postfix = OrderedDict([('iou',out_iou)])
    tbar.set_postfix(postfix)
    tbar.update(1)
tbar.close
label_list = np.array([0,1,2,3,4,5,6,7,8,9])
total_idx = np.arange(0, len(label_list))
pred_idx = [i for i in total_idx]
dice_score_pred = np.asarray(dice_score_list)
iou_score_pred = np.asarray(iou_score_list)
iou_score_pred = iou_score_pred # [:,pred_idx] #??????????????????????????????????????
print('Mean of dice score:'+str(np.mean(dice_score_pred)))
print('Mean of iou score:'+str(np.mean(iou_score_pred)))
  
