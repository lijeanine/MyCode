# pay attention to the differet use between validation and testing of dice.py
#function of dice: dice = 2*(A & B)/(|A| + |B|)
def score_perclass(vol_output, ground_truth, num_classes):# [validation input:N,C,H,W](sampled batch)  [testing input:C,H,W](whole 3D image)
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

#----------------------------------------------------in validation---------------------------------------------------------------------
#focus on the structure
def validate(config, val_loader, model, criterion): #copy from UNext train.py validate()
    avg_meters = {'loss': AverageMeter()}#ori:add :,'iou': AverageMeter(),'dice': AverageMeter()

    # switch to evaluate mode
    model.eval()
    batch_dice_list=[]
    batch_iou_list=[]
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        #for input, target, _ in val_loader:
        for sample_val in val_loader:
            input = sample_val['image_stack'].cuda()
            target = sample_val['label'].cuda()

            # compute output
            if config['deep_supervision']: #set False
                outputs = model(input)
                _, batch_output = torch.max(outputs, dim=1)
                loss = 0
                dice,iou = score_perclass(batch_output,target,10)
                dice = dice.cpu().numpy()
                iou = iou.cpu().numpy()
                batch_dice_list.append(dice)
                batch_iou_list.append(iou)
                for output in outputs:
                    loss += criterion(output, target.long())
                loss /= len(outputs)
                #ori:iou,dice = iou_score(outputs[-1], target)
            else:
                output = model(input)
                _, batch_output = torch.max(output, dim=1)
                loss = 0
                dice,iou = score_perclass(batch_output,target,10)
                dice = dice.cpu().numpy()
                iou = iou.cpu().numpy()
                batch_dice_list.append(dice)
                batch_iou_list.append(iou) #len = 800,(800 iterations),batch_iou_list[0].shape=[10,]
                loss = criterion(output, target.long())
                #ori:iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            #ori:avg_meters['iou'].update(iou, input.size(0))
            #ori:avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                #ori:('iou', avg_meters['iou'].avg),
                #ori:('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
        dice_score_arr = np.asarray(batch_dice_list)
        iou_score_arr = np.asarray(batch_iou_list)
        avg_dice_score = np.mean(dice_score_arr)
        avg_iou_score = np.mean(iou_score_arr)
        print('Validation avg_dice_score:',avg_dice_score)
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_iou_score),
                        ('dice', avg_dice_score)])





#----------------------------------------------------in testing------------------------------------------------------------------------
def testing(args,model): #copy from UNext val.py Validation()
    #args = parse_args()
    model = model.cuda()
    # Data loading code
    model.load_state_dict(torch.load(args.resume))
    model.eval()
    test_data = LoadMRIData(args.data_dir, args.data_list, 'test', args.num_class, num_slices=args.num_slices, se_loss = False, Encode3D=args.encode3D)
    test_loader = DataLoader(test_data, batch_size = 1, shuffle = False, num_workers = args.workers, pin_memory=True)
    tbar = tqdm(test_loader, desc='test_loader') #\r
    volume_dice_score_list = []
    volume_iou_score_list = []
    batch_size = 2   #args.test_batch_size
    with torch.no_grad():
        for ind, sample_batched in enumerate(tbar):
            volume = sample_batched['image_3D'].type(torch.FloatTensor)#[1,256,256,256]
            labelmap = sample_batched['label_3D'].type(torch.LongTensor)
            volume = torch.squeeze(volume) #[256,256,256]
            labelmap = torch.squeeze(labelmap)
            sample_name = sample_batched['name']
            
            if args.cuda:
                volume, labelmap = volume.cuda(), labelmap.cuda()
            
            z_ax, x_ax, y_ax = np.shape(volume)
            
            volume_prediction = []
            for i in range(0, len(volume), batch_size): #len(volume) = 256 batch_size=2

                if i<=int( args.num_slices*2+1):
                    image_stack0 = volume[0:int( args.num_slices*2+1),:,:][None]
                    image_stack1 = volume[1:int( args.num_slices*2+2),:,:][None]
                elif i >=z_ax-int( args.num_slices*2+1):
                    image_stack0 = volume[z_ax-int( args.num_slices*2+2):-1,:,:][None]
                    image_stack1 = volume[z_ax-int( args.num_slices*2+1):,:,:][None]
                else:
                    image_stack0 = volume[i- args.num_slices:i+ args.num_slices+1,:,:][None]
                    image_stack1 = volume[i- args.num_slices+1:i+ args.num_slices+2,:,:][None]
                
                image_3D = torch.cat((image_stack0, image_stack1), dim =0) #[2,11,256,256]
                
                outputs =  model(image_3D)
                #pred = outputs[0]#outputs[2]:[2,2,256,256]
                                 #outputs[0]:[2,10,256,256],10 labels
                _value, batch_output = torch.max(outputs, dim=1) #label:[2,10,256,256] -> [2,256,256]
                #batch_output = pred[:,0,:,:]
                volume_prediction.append(batch_output)
            
            #volume and label are both CxHxW
            volume_prediction = torch.cat(volume_prediction)
            volume_dice_score, volume_iou_score= score_perclass(torch.narrow(volume_prediction,0,0,z_ax), labelmap,  args.num_class)
            
            volume_dice_score = volume_dice_score.cpu().numpy()
            volume_dice_score_list.append(volume_dice_score)
            tbar.set_description('Validate Dice Score: %.3f' % (np.mean(volume_dice_score)))
            
            volume_iou_score = volume_iou_score.cpu().numpy()
            volume_iou_score_list.append(volume_iou_score)
            
            
            #########################save output to directory##################################
            savedir_pred = os.path.join( args.save_dir,'pred')
            if not os.path.exists(savedir_pred):
                os.makedirs(savedir_pred)
            volume_prediction = volume_prediction.cpu().numpy().astype(np.uint8)
            volume_prediction = np.transpose(volume_prediction, (1,2,0))
            nib_pred = nib.Nifti1Image(volume_prediction, affine=np.eye(4))
            nib.save(nib_pred, os.path.join(savedir_pred, sample_name[0]+'.nii.gz'))
            #########################save output to directory##################################
        
        del volume_prediction
        #####################################use 134 classes for evaluation####################################
        dice_score_arr = np.asarray(volume_dice_score_list)
        iou_score_arr = np.asarray(volume_iou_score_list)
        
        label_list = np.array([0,1,2,3,4,5,6,7,8,9])
        total_idx = np.arange(0, len(label_list))
        #ignore = np.array([42, 43, 64, 69])
        
        #valid_idx = [i+1 for i in total_idx if label_list[i] not in ignore]
        #valid_idx = [0] + valid_idx
        valid_idx = [i for i in total_idx ] #[0,1,2,3,4,5,6,7,8,9]
        
        dice_socre_vali = dice_score_arr[:,valid_idx]
        avg_dice_score = np.mean(dice_socre_vali)
        std_dice_score = np.std(dice_socre_vali)
        
        iou_score_vali = iou_score_arr[:,valid_idx]
        avg_iou_score = np.mean(iou_score_vali)
        std_iou_score = np.std(iou_score_vali)
        ##########################################################################################################
        
        print('Validation:')
        print("Mean of dice score : " + str(avg_dice_score))
        print("Std of dice score : " + str(std_dice_score))
        print("Mean of iou score : " + str(avg_iou_score))
        print("Std of dice score : " + str(std_iou_score))











