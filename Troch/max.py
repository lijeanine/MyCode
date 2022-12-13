pre_mask = pre_mask.argmax(dim=1)  # input:pre_mask.shape = [8,10,256,256] output:pre_mask.shape = [8,256,256]
_value, batch_output = torch.max(pred, dim=1) #label:[2,10,256,256] -> [2,256,256]
