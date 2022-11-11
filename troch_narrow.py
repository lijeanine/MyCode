x_cat = torch.narrow(x_cat, 2, self.pad, H)    #x_cat.shape=[4,160,16,20] slices [2,17] in dim 2 
x_s = torch.narrow(x_cat, 3, self.pad, W)
------------------------------------------------------------
x_s = x_cat[:, :, pad:-pad, pad:-pad] # same function as torch.narrow
