#-----------------------------------------------------stat and torch method---------------------------------------------------------------
from torchstat import stat
from Modules.backbone import Backbone
if __name__=='__main__':
    model = Backbone(10,5)
    total = sum([param.nelement() for param in model.parameters()])#torch method
    print("Number of parameters: %.2fM" % (total/1e6))
    stat(model,(3,256,256))#show all params, torchstat method for simple model
#-------------------------------------------------------thop method-----------------------------------------------------------------------
from thop import profile
'''
from torchvision.models import resnet50
model = resnet50()
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))
print("FLOPs=", str(flops/1e9) + '{}'.format("G"))
print("params=", str(params/1e6) + '{}'.format("M"))
'''
import torch
import torch.nn as nn
from Modules.backbone import Backbone
class func(nn.Module):
    def __init__(self,object):
        super(func,self).__init__()
        self.model = object
    def forward(self,a):
        T1 = torch.randn(4,11,256,256).float().cuda()#add .cuda() for AS-MLP shift_cuda
        T2 = torch.randn(4,11,256,256).float().cuda()
        out = self.model(T1,T2)
        return out
model = Backbone(10,5).cuda()
use_model = func(model)
a = torch.randn(4,11,256,256).float()
print(a.shape)
flops,params = profile(use_model,inputs=(a,))#when only one element, ',' is required.
print("FLOPs=",str(flops/1e9)+'{}'.format("G"))
print("params=",str(params/1e6)+'{}'.format("M"))
