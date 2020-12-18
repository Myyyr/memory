import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import revtorch.revtorch as rv
import random


class softmax(nn.Module):
    def __init__(self, dim):
        super(softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim = self.dim)



id = random.getrandbits(64)
def convert_byte(v):
    units = {'Bytes':1,'KB':1e-3, 'MB':1e-6, 'GB':1e-9}
    tmp = 'Bytes'
    for k in list(units.keys()):
        if int(v*units[k]) == 0:
            return v*units[tmp], tmp
        tmp = k
    return v*units[tmp], tmp

def prt_mem(txt):
    a = torch.cuda.max_memory_allocated()
    b = torch.cuda.memory_allocated()
    print(txt,':' ,convert_byte(a) , convert_byte(b))
    #return a,b

class ResidualInner(nn.Module):
    def __init__(self, channels, groups):
        super(ResidualInner, self).__init__()
        # self.gn = nn.BatchNorm3d(channels)
        self.gn = nn.GroupNorm(groups, channels)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.gn(self.conv(x)), inplace=True)
        return x

def makeReversibleSequence(channels):
    innerchannels = channels // 2
    groups = 2 if innerchannels > 1 else 1 #channels[0] // 2
    # print("chan, groups" ,channels, groups)
    fBlock = ResidualInner(innerchannels, groups)
    gBlock = ResidualInner(innerchannels, groups)
    #gBlock = nn.Sequential()
    return rv.ReversibleBlock(fBlock, gBlock)

def makeReversibleComponent(channels, blockCount):
    modules = []
    for i in range(blockCount):
        modules.append(makeReversibleSequence(channels))
    return rv.ReversibleSequence(nn.ModuleList(modules))

def getchannelsAtIndex(index, channels):
    if index < 0: index = 0
    if index >= len(channels): index = len(channels) - 1
    return channels[index]

class EncoderModule(nn.Module):
    def __init__(self, inchannels, outchannels, depth, downsample=True):
        super(EncoderModule, self).__init__()
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool3d(2)
            self.conv = nn.Conv3d(inchannels, outchannels, 1)
        self.reversibleBlocks = makeReversibleComponent(outchannels, depth)

    def forward(self, x):
        if self.downsample:
            # x = F.max_pool3d(x, 2)
            x = self.pool(x)
            x = self.conv(x) #increase number of channels
        x = self.reversibleBlocks(x)
        return x

class Interpolate(nn.Module):
    def __init__(self, scale_factor = 2, mode = "trilinear", align_corners = False):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor 
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)



class DecoderModule(nn.Module):
    def __init__(self, inchannels, outchannels, depth, upsample=True):
        super(DecoderModule, self).__init__()
        self.reversibleBlocks = makeReversibleComponent(inchannels, depth)
        self.upsample = upsample
        if self.upsample:
            self.conv = nn.Conv3d(inchannels, outchannels, 1)
            self.interpolate = Interpolate()

    def forward(self, x, shape):
        x = self.reversibleBlocks(x)
        if self.upsample:
            x = self.conv(x)
            x = self.interpolate(x)
        for i in range(1,4):
            # print("#" ,x.shape,shape)
            if x.shape[-i] != shape[-i]:

                tup = [0,0,0,0,0,0]
                n_tmp = abs(x.shape[-i] - shape[-i])
                tup[i*2 -1] = n_tmp

                x = F.pad(x, tuple(tup), 'constant')
                # print("##", n_tmp, x.shape)
        return x

class RevUnet3D(nn.Module):
    def __init__(self, inchannels ,channels, out_size, interpolation = None):
        super(RevUnet3D, self).__init__()
        depth = 1
        self.levels = len(channels)

        self.firstConv = nn.Conv3d(inchannels, channels[0], 3, padding=1, bias=False)
        #self.dropout = nn.Dropout3d(0.2, True)
        

        #create encoder levels
        encoderModules = []
        for i in range(self.levels):
            encoderModules.append(EncoderModule(getchannelsAtIndex(i - 1, channels), getchannelsAtIndex(i, channels), depth, i != 0))
        self.encoders = nn.ModuleList(encoderModules)

        #create decoder levels
        decoderModules = []
        for i in range(self.levels):
            decoderModules.append(DecoderModule(getchannelsAtIndex(self.levels - i - 1, channels), getchannelsAtIndex(self.levels - i - 2, channels), depth, i != (self.levels -1)))
        self.decoders = nn.ModuleList(decoderModules)

        self.lastConv = nn.Conv3d(channels[0], out_size, 1, bias=True)

        self.softmax = softmax(1)

        self.interpolation = interpolation
        if self.interpolation != None:
            self.interpolation = nn.Upsample(size = interpolation, mode = "trilinear")



    def forward(self, x):
        # tibo_in_shape = x.shape[-3:]
        # print("x.shape :",x.shape)
        prt_mem('start')
        x = self.firstConv(x)
        prt_mem('firstConv')
        #x = self.dropout(x)

        inputStack = []
        shapes = [x.shape]
        # print("level :", -1, " x.shape :",x.shape)
        for i in range(self.levels):
            
            x = self.encoders[i](x)
            # prt_mem('encoder.'+str(i))
            shapes.append(x.shape)
            # print("level :", i, " x.shape :",x.shape)
            if i < self.levels - 1:
                inputStack.append(x)

        for i in range(self.levels):
            
            x = self.decoders[i](x, shapes[-(i+2)])
            # prt_mem('decoder.'+str(i))
            # print("level :", i, " x.shape :",x.shape)
            if i < self.levels - 1:
                x = x + inputStack.pop()

        x = self.lastConv(x)
        # prt_mem('lastConv')
        x = self.softmax(x)
        # if self.interpolation != None:
        #     xi = self.interpolation(x)
        #     # prt_mem('interpolation')
        #     return xi, x
        # #x = torch.sigmoid(x)
        return self.interpolation(x)

    @staticmethod
    def apply_argmax_softmax(pred):
        pred = F.softmax(pred, dim=1)
        return pred