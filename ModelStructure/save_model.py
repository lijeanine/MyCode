#model saving with averiging method for different model`s output
# save weights(.pth) and log file(.csv)
from collections import OrderedDict
import pandas as pd 
def main():
    ...
    model=my_model.cuda()
    log = OrderedDict([
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
    for epoch in range(config['epochs']):
        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)
        print('loss %.4f - val_loss %.4f ' #ori:- iou %.4f - val_iou %.4f
              % (train_log['loss'], val_log['loss']))#ori:add train_log['iou'], val_log['iou']

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        #ori:log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1
        
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0
        
        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
