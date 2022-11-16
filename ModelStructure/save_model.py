#model saving with averiging method for different model`s output
# save weights(.pth) and log file(.csv)
from collections import OrderedDict
import pandas as pd 
def main():
    ...
    model=my_model.cuda()
    log = OrderedDict([   #2.log:build log dict
    ('epoch', []),
    ('lr', []),
    ('loss', []),
    #ori:('iou', []),
    ('val_loss', []),
    ('val_iou', []),
    ('val_dice', []),
     ])
    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']): #1.model:save model for every epoch
        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion) #1.1 metrics of model on validation set 
        print('loss %.4f - val_loss %.4f ' #ori:- iou %.4f - val_iou %.4f
              % (train_log['loss'], val_log['loss']))#ori:add train_log['iou'], val_log['iou']

        log['epoch'].append(epoch) #2.1 save validation log for every epoch
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        #ori:log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %     #2.2 show log by csv
                                 config['name'], index=False)

        trigger += 1
        
        if val_log['iou'] > best_iou:#1.2 judge and if its true,save weights.
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0
        
        # 1.3 early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
