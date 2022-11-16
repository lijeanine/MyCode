def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter()}#ori:add :,'iou': AverageMeter(),'dice': AverageMeter()
    #1.switch to evaluate mode
    model.eval() 
    batch_dice_list=[]
    batch_iou_list=[]
    with torch.no_grad():#2.maintain gradient
        pbar = tqdm(total=len(val_loader))
        #for input, target, _ in val_loader:
        for sample_val in val_loader: #3.start validation 
            input = sample_val['image_stack'].cuda()
            target = sample_val['label'].cuda()

            #3.1 compute output
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
