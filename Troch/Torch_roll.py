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
class SplitAttention(nn.Module):
    def __init__(self, channel = 512, k = 3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias = False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias = False)
        self.softmax = nn.Softmax(1)
    
    def forward(self,x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)          #bs,k,n,c
        a = torch.sum(torch.sum(x_all, 1), 1)       #bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  #bs,kc
        hat_a = hat_a.reshape(b, self.k, c)         #bs,k,c
        bar_a = self.softmax(hat_a)                 #bs,k,c
        attention = bar_a.unsqueeze(-2)             # #bs,k,1,c
        out = attention * x_all                     # #bs,k,n,c
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out
