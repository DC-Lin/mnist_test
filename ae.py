import torch
from torch import nn
import visdom
viz=visdom.Visdom()
x_train=0
y_train=0
class AE(nn.Module):
    def __init__(self):
        super(AE,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.Linear(16,64),
            nn.ReLU(),
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Sigmoid()
        )
    def forward(self, x):
        x=x.view(x.size(0),784)
        x=self.encoder(x)
        y=x
        viz.images(x,nrow=8,win='middle_x',opts=dict(title='middle_x'))
        x=self.decoder(x)
        x=x.view(x.size(0),1,28,28)
        return x

class mnist_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(mnist_conv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=2,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(ch_out,64,kernel_size=2,stride=2,padding=0),
            nn.ReLU()
        )
