##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Created by: Yuemeng Li
#BE department, University of Pennsylvania
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from Modules.backbone import Backbone
import torch
import argparse
import os
import numpy as np
from data_utils.MRIloader import LoadMRIData
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils.lr_scheduler import LR_Scheduler
import Modules.lossmodule as lm
from torch.nn import DataParallel
import nibabel as nib
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

####################Parameters to define##############################
RESUME_PATH = None #directory you want to resume the model
SAVE_DIR = './save_model/demo/' #directory you want to save the model
NUM_CLASS = 10 #for MALC-coarse grained use 28, for MALC-fine grained use 139
TWO_STAGES = True #False use end-to-end training
RESUME_PRETRAIN = False #check line 104
######################################################################

DATA_DIR = '/opt/data/private/ACEnet_dhcp/resampled_dhcp/'
DATA_LIST = '/opt/data/private/ACEnet_dhcp/datasets_dhcp/'



class Solver():
    def __init__(self, args, num_class):
        self.args = args
        self.num_class = num_class

        #load data
        train_data = LoadMRIData(args.data_dir, args.data_list, 'train', num_class, num_slices=args.num_slices, se_loss = args.se_loss, use_weight = args.use_weights, Encode3D=args.encode3D)
        self.train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = args.workers, pin_memory=True)
        
        test_data = LoadMRIData(args.data_dir, args.data_list, 'test', num_class, num_slices=args.num_slices, se_loss = False, Encode3D=args.encode3D)
        self.test_loader = DataLoader(test_data, batch_size = 1, shuffle = False, num_workers = args.workers, pin_memory=True)
        
        model = Backbone(num_class, args.num_slices)
        
        ####################################################################################
        #set optimizer for different training strategies
        if args.two_stages:
            optimizer = torch.optim.SGD([{'params': model.encode3D1.parameters()},
                                     {'params': model.encode3D2.parameters()},
                                     {'params': model.encode3D3.parameters()},
                                     {'params': model.encode3D4.parameters()},
                                     {'params': model.bottleneck3D.parameters()},
                                     {'params': model.decode4.parameters()},
                                     {'params': model.decode3.parameters()},
                                     {'params': model.decode2.parameters()},
                                     {'params': model.decode1.parameters()},
                                     {'params': model.encmodule.parameters()},
                                     {'params': model.conv6.parameters()},
                                     #new added parameters
                                     {'params': model.decode0.parameters(), 'lr': args.lr},
                                     {'params': model.conv7.parameters(), 'lr': args.lr},
                                     {'params': model.conv8.parameters(), 'lr': args.lr},
                                     ],
                                    lr=1e-7, momentum=0.9, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=0.9, weight_decay=args.weight_decay)
        ####################################################################################
                
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))
        #define loss
        self.criterion = lm.CombinedLoss(se_loss = args.se_loss)
        
        self.model, self.optimizer = model, optimizer
        
         # Using cuda
        if args.cuda:
            self.model = DataParallel(self.model).cuda()
            self.criterion = self.criterion.cuda()
        
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            
            if args.resume_pretrain:
                args.start_epoch = checkpoint['epoch']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_pred = checkpoint['best_pred']
            
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            
            if args.cuda:
                self.model.module.load_state_dict(model_dict)
            else:
                self.model.load_state_dict(model_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))


    def train(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        for i, sample_batched in enumerate(tbar):
            
            target = sample_batched['label'].type(torch.LongTensor)
            skull = sample_batched['skull'].type(torch.LongTensor)
            
            if self.args.cuda:
                target, skull = target.cuda(), skull.cuda()
            
            
            image_3D = sample_batched['image_stack'].type(torch.FloatTensor)
            image_3D = image_3D.cuda() #[4,11,256,256]
            
            se_gt = None
            if self.args.se_loss:
                se_gt = sample_batched['se_gt'].type(torch.FloatTensor)
                if self.args.cuda:
                    se_gt = se_gt.cuda()
            
            weights = None
            if self.args.use_weights:
                weights = sample_batched['weights'].type(torch.FloatTensor)
                if self.args.cuda:
                    weights = weights.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            
            outputs = self.model(image_3D) 
            loss = self.criterion(outputs, target, skull, se_gt, weight = weights) #in fact we don't use skull to caculate loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            
            del target, skull, image_3D
            
        print("==== Epoch [" + str(epoch) + " / " + str(self.args.epochs) + "] DONE ====")
        print('Loss: %.3f' % train_loss)
        
        if self.args.no_val:
            # save checkpoint every epoch
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, os.path.join(self.args.save_dir, 'checkpoint_%03d.pth.tar' % epoch))
            print("save model on epoch %d" % epoch)
            
        
    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        volume_dice_score_list = []
        volume_iou_score_list = []
        batch_size = self.args.test_batch_size
        
        with torch.no_grad():
            for ind, sample_batched in enumerate(tbar):
                volume = sample_batched['image_3D'].type(torch.FloatTensor)
                labelmap = sample_batched['label_3D'].type(torch.LongTensor)
                volume = torch.squeeze(volume)
                labelmap = torch.squeeze(labelmap)
                sample_name = sample_batched['name']
                
                if self.args.cuda:
                    volume, labelmap = volume.cuda(), labelmap.cuda()
                
                z_ax, x_ax, y_ax = np.shape(volume)
                
                volume_prediction = []
                skull_prediction = []
                for i in range(0, len(volume), batch_size):

                    if i<=int(self.args.num_slices*2+1):
                        image_stack0 = volume[0:int(self.args.num_slices*2+1),:,:][None]
                        image_stack1 = volume[1:int(self.args.num_slices*2+2),:,:][None]
                    elif i >=z_ax-int(self.args.num_slices*2+1):
                        image_stack0 = volume[z_ax-int(self.args.num_slices*2+2):-1,:,:][None]
                        image_stack1 = volume[z_ax-int(self.args.num_slices*2+1):,:,:][None]
                    else:
                        image_stack0 = volume[i-self.args.num_slices:i+self.args.num_slices+1,:,:][None]
                        image_stack1 = volume[i-self.args.num_slices+1:i+self.args.num_slices+2,:,:][None]
                    
                    image_3D = torch.cat((image_stack0, image_stack1), dim =0)
                    
                    outputs = self.model(image_3D)
                    pred = outputs[0]
                    #skull_pred = outputs[2]
                    
                    _, batch_output = torch.max(pred, dim=1)
                   # _, skull_output = torch.max(skull_pred, dim=1)
                    volume_prediction.append(batch_output)
                    #skull_prediction.append(skull_output)
                
                #volume and label are both CxHxW
                volume_prediction = torch.cat(volume_prediction)
                
                #dice and iou evaluation
                volume_dice_score, volume_iou_score= score_perclass(volume_prediction, labelmap, self.num_class)
                
                volume_dice_score = volume_dice_score.cpu().numpy()
                volume_dice_score_list.append(volume_dice_score)
                tbar.set_description('Validate Dice Score: %.3f' % (np.mean(volume_dice_score)))
                
                volume_iou_score = volume_iou_score.cpu().numpy()
                volume_iou_score_list.append(volume_iou_score)

                ####################save output for visualization##################################
                visual = False
                if visual:
                    savedir_pred = os.path.join(self.args.save_dir,'pred')
                    if not os.path.exists(savedir_pred):
                        os.makedirs(savedir_pred)
                    volume_prediction = volume_prediction.cpu().numpy().astype(np.uint8)
                    volume_prediction = np.transpose(volume_prediction, (1,2,0))
                    nib_pred = nib.Nifti1Image(volume_prediction, affine=np.eye(4))
                    nib.save(nib_pred, os.path.join(savedir_pred, sample_name[0]+'.nii.gz'))
                ####################save output for visualization##################################
                
            
            del volume_prediction
            
            dice_score_arr = np.asarray(volume_dice_score_list)
            iou_score_arr = np.asarray(volume_iou_score_list)
            
            ####################################save best model for dice###################
            if self.args.num_class is 139:
                label_list = np.array([4,  11,  23,  30,  31,  32,  35,  36,  37,  38,  39,  40,
                           41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  55,
                           56,  57,  58,  59,  60,  61,  62,  63,  64,  69,  71,  72,  73,
                           75,  76, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112,
                          113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                          128, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
                          143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
                          156, 157, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
                          171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
                          184, 185, 186, 187, 190, 191, 192, 193, 194, 195, 196, 197, 198,
                          199, 200, 201, 202, 203, 204, 205, 206, 207])
                total_idx = np.arange(0, len(label_list))
                ignore = np.array([42, 43, 64, 69])
                
                valid_idx = [i+1 for i in total_idx if label_list[i] not in ignore]
                valid_idx = [0] + valid_idx
                
                dice_socre_vali = dice_score_arr[:,valid_idx]
                iou_score_vali = iou_score_arr[:,valid_idx]
            else:
                dice_socre_vali = dice_score_arr
                iou_score_vali = iou_score_arr
            ####################################save best model for dice###################
            
            avg_dice_score = np.mean(dice_socre_vali)
            std_dice_score = np.std(dice_socre_vali)
            avg_iou_score = np.mean(iou_score_vali)
            std_iou_score = np.std(iou_score_vali)
            print('Validation:')
            print("Mean of dice score : " + str(avg_dice_score))
            print("Std of dice score : " + str(std_dice_score))
            print("Mean of iou score : " + str(avg_iou_score))
            print("Std of dice score : " + str(std_iou_score))
            
            if avg_dice_score>self.best_pred:
                np.save(self.args.save_dir + 'dice_score.npy', dice_score_arr)
                np.save(self.args.save_dir + 'iou_score.npy', iou_score_arr)
                self.best_pred = avg_dice_score
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, os.path.join(self.args.save_dir, 'checkpoint_%03d.pth.tar' % epoch))
                print("save model on epoch %d" % epoch)
        

def score_perclass(vol_output, ground_truth, num_classes):
    dice_perclass = torch.zeros(num_classes)
    iou_perclass = torch.zeros(num_classes)
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        inter = torch.sum(torch.mul(GT, Pred))
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union))
        iou_perclass[i] = torch.div(inter, union-inter)
    return dice_perclass, iou_perclass
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (dcriterionefault: 8)')
    parser.add_argument('--resume', type=str, default=RESUME_PATH,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default=SAVE_DIR, type=str, metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='directory to load data')
    parser.add_argument('--data-list', type=str, default=DATA_LIST,
                        help='directory to read data list')
    parser.add_argument('--encode3D', action='store_true', default=True,
                        help='directory to read data list')
    parser.add_argument('--se-loss', action='store_false', default=True,
                        help='apply se classification loss')
    parser.add_argument('--use-weights', action='store_false', default=False,
                        help='apply class weights for 2DCE loss')
    parser.add_argument('--two-stages', action='store_false', default=TWO_STAGES,
                        help='apply two stages training')
    parser.add_argument('--resume-pretrain', action='store_false', default=RESUME_PRETRAIN,
                        help='for training from scratch, set False, for training resume from model, set True')
    # training hyper params
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b-train', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-b-test', '--test-batch-size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-num-slices', '--num-slices', default=5, type=int,
                        metavar='N', help='slice thickness for spatial encoding')
    parser.add_argument('-num-class', '--num-class', default=NUM_CLASS, type=int,
                        metavar='N', help='number of classes for segmentation')
    # cuda, seed and loggingevaluator
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='check whether to use cuda')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # optimizer params
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    print(args)
    solver = Solver(args, args.num_class)
    print('Starting Epoch:', solver.args.start_epoch)
    print('Total Epoches:', solver.args.epochs)

    for epoch in range(solver.args.start_epoch, solver.args.epochs):
        solver.train(epoch)
        if not solver.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            solver.validation(epoch)
    
if __name__ == '__main__':
    main()
    
    
