x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))] #xs is tuple,len(xs)=5,xs[i].shape=[4,32,20,20]
x_cat = torch.cat(x_shift, 1)
x_cat = torch.narrow(x_cat, 2, self.pad, H) #self.pad=2
x_s = torch.narrow(x_cat, 3, self.pad, W)
#x_shift is list,len(x_shift)=5,x_shift[i].shape=[4,32,20,20]
#for i in [-2,-1,0,-1,2]:shift = i 
#for i in (0,5):x_c.shape=[4,32,20,20] 
-------------------------------------------------------------------------------------------------------------------------------------------------------------

def spatial_shift1(x):  # same function as torch.roll
    b,w,h,c = x.size()
    x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
    x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
    x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
    x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
    return x

def spatial_shift2(x):
    b,w,h,c = x.size()
    x[:,:,1:,:c//4] = x[:,:,:h-1,:c//4]
    x[:,:,:h-1,c//4:c//2] = x[:,:,1:,c//4:c//2]
    x[:,1:,:,c//2:c*3//4] = x[:,:w-1,:,c//2:c*3//4]
    x[:,:w-1,:,3*c//4:] = x[:,1:,:,3*c//4:]
    return x

