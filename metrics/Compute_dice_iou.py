# compute dice and iou of predicition with nii.gz format
import nibabel as nib
import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
pred_path = './UNext_SwinDHCP/pred/'
label_path = './9labels_SwinDHCP/labels/'
pred_names = os.listdir(pred_path)
# file for saving
tbar = tqdm(total=len(pred_names))
dice_score_list=[]
iou_score_list=[]
def score_perclass(vol_output, ground_truth, num_classes):
    dice_perclass = torch.zeros(num_classes)
    iou_perclass = torch.zeros(num_classes)
    vol_output = torch.from_numpy(vol_output) #convert ndarray to tensor
    ground_truth = torch.from_numpy(ground_truth)#convert ndarray to tensor
    x,y,z = np.shape(ground_truth) #
    vol_output = torch.narrow(vol_output,2,0,z)
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        inter = torch.sum(torch.mul(GT, Pred))
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union))
        iou_perclass[i] = torch.div(inter, union-inter)
    return dice_perclass.numpy(), iou_perclass.numpy()  #convert tensor to ndarray

for pred_name in pred_names:
    #1.load pred and label
    label_name = str(pred_name).split('.')[0]+'_glm.nii.gz'
    pred = nib.load(pred_path + pred_name).get_data()
    pred = np.array(pred)
    label = nib.load(label_path + label_name).get_data()
    label = np.array(label)
    #2.compute dice and iou
    out_dice, out_iou = score_perclass(pred, label, 10)
    #3.show value of metrics
    dice_score_list.append(out_dice)
    iou_score_list.append(out_iou)
    tbar.set_description('Predict Dice Score:%.3f'%(np.mean(out_dice))) #out_dice is a list of length with 10
    postfix = OrderedDict([('iou',np.mean(out_iou))])
    tbar.set_postfix(postfix)
    tbar.update(1)
tbar.close

label_list = np.array([0,1,2,3,4,5,6,7,8,9])
total_idx = np.arange(0, len(label_list))
pred_idx = [i for i in total_idx]
dice_score_pred = np.asarray(dice_score_list)
iou_score_pred = np.asarray(iou_score_list)

iou_score_pred = iou_score_pred#[:,pred_idx] 
#we use all of the 10 value in dimesion 1, so whether to use pred_idx or not,both are ok. 
print('Mean of metrics:')
print('Mean of dice score:'+str(np.mean(dice_score_pred)))
print('Mean of iou score:'+str(np.mean(iou_score_pred)))
