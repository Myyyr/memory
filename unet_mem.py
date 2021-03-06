import torch
from models.unet_3D import unet_3D
from models.unetUtils import Pad
import models
import numpy as np




def get_mod_details(model):
    ret = {'name' : [], 'layer' : []}
    for name, layer in model.named_modules():
        #print(name, layer, type(layer))
        #print(name, layer)
        # if name != "" and (isinstance(layer, models.unet_3D.softmax) or isinstance(layer, torch.nn.modules.upsampling.Upsample) or isinstance(layer, torch.nn.Conv3d) or isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.GroupNorm) or isinstance(layer, torch.nn.MaxPool3d) or isinstance(layer, models.unetUtils.Pad)):
        if name != "" and (isinstance(layer, models.unet_3D.softmax) or isinstance(layer, torch.nn.modules.upsampling.Upsample) or isinstance(layer, torch.nn.Conv3d)  or isinstance(layer, torch.nn.GroupNorm) or isinstance(layer, torch.nn.MaxPool3d) or isinstance(layer, models.unetUtils.Pad)):
            ret['name'].append(name)
            ret['layer'].append(layer)
    return ret



def get_activations_shapes_as_dict(layers, x):
    shapes = {'input':x.shape}
    tmp = []
    up = -1
    with torch.no_grad():
        for n, l in zip(layers['name'], layers['layer']):

            # print(n)
            
            if 'max' in n:
                shapes[n+'_res'] = x.shape
                tmp.append(x)
                x = l(x)
                shapes[n] = x.shape
            elif 'pad' in n:
                inputs1 = tmp[up]
                offset = x.size()[2] - inputs1.size()[2]
                padding = 2 * [offset // 2, offset // 2, 0]
                x, y = l(inputs1, x, padding)
                x = torch.cat([x,y],1)
                shapes[n] = y.shape
                up -= 1    
            else:
                x = l(x)
                shapes[n] = x.shape



            # print(n, x.shape)
            if 'center.relu2' in n:
                shapes[n+str('_res')] = x.shape
    return shapes

def forward_memory_cosumption_with_peak(shapes, float_types='float'):
    cur_m = 0
    all_types = {'float':4, 'double':8, 'half':2}
    
    enc, dec = -1, -1
    for k in list(shapes.keys()):
        s = shapes[k]
        cur_m += np.prod(s)*all_types[float_types]

    return cur_m

def labels_mem(y,floatt = 'float'):
    return np.prod(y.shape)*flt(floatt)


def model_memory(mod, floatt = 'float'):
    return sum(p.numel()*flt(floatt) for p in mod.parameters())





def convert_byte(v):
    units = {'Bytes':1,'KB':1e-3, 'MB':1e-6, 'GB':1e-9}
    tmp = 'Bytes'
    for k in list(units.keys()):
        if int(v*units[k]) == 0:
            return v*units[tmp], tmp
        tmp = k
    return v*units[tmp], tmp

def flt(t):
    return {'float':4, 'double':8, 'half':2, 'short':4}[t]


def validat_arg_memory(arg, floatt = 'short'):
    return np.prod(arg.shape)*flt(floatt)


def main():
    # print(convert_byte(1000*1000*1000*3*4))
    # print(convert_byte(1000*1000*1000*4*4))
    # print(convert_byte(1000*1000*1000*5*4))
    # exit(0)
    inchan = 1
    chanscale = 2
    chans = [i//chanscale for i in [64, 128, 256, 512, 1024]]
    outsize = 14
    # interp = (512,512,198)
    mod = unet_3D(chans, n_classes=outsize, in_channels=inchan, interpolation = None)

    layers = get_mod_details(mod)

    fact = 0.1
    # s = (80,80,32)
    # s = (112,112,48)
    s = (256,256,112)
    # x = torch.from_numpy(np.random.rand(1,1,int(round(512*fact)),int(round(512*fact)),int(round(198*fact)))).float()
    # y = torch.from_numpy(np.random.rand(1,outsize,int(round(512*fact)),int(round(512*fact)),int(round(198*fact)))).float()
    # argmax = torch.from_numpy(np.random.rand(1,outsize,int(round(512*fact)),int(round(512*fact)),int(round(198*fact)))).float()

    x = torch.from_numpy(np.random.rand(1,1,s[0], s[1], s[2])).float()
    y = torch.from_numpy(np.random.rand(1,outsize,s[0], s[1], s[2])).float()
    argmax = torch.from_numpy(np.random.rand(1,outsize,s[0], s[1], s[2])).float()
    acts = get_activations_shapes_as_dict(layers, x)


    

    mod_m = model_memory(mod)
    lab_m = labels_mem(y)
    argm_m = validat_arg_memory(argmax)
    inp_m = labels_mem(x)
    cur_m = forward_memory_cosumption_with_peak(acts) - inp_m
    back_m = chans[0]*np.prod(s)*4*2 - outsize*np.prod(s)*4*3

    # print(convert_byte(cur_m*2 + mod_m + lab_m))
    # print(convert_byte(lab_m))
    # print(convert_byte(cur_m),convert_byte(mod_m),convert_byte(lab_m))
    # print(convert_byte(mod_m))
    # print(convert_byte(mod_m+ labels_mem(x)))
    # print(convert_byte(mod_m+ labels_mem(x) + lab_m))
    # print(convert_byte(mod_m+ labels_mem(x) + lab_m + (cur_m + lab_m)))
    print('model :', convert_byte(mod_m))
    print('input :', convert_byte(mod_m+inp_m))
    print('label :', convert_byte(mod_m+inp_m + lab_m))
    print('forwa :', convert_byte(mod_m+inp_m + lab_m + cur_m))
    print('backw :', convert_byte(mod_m+inp_m + lab_m + cur_m + back_m))
    print('max   :', convert_byte(max(mod_m+inp_m + lab_m + cur_m, mod_m+inp_m + lab_m + cur_m + back_m - 2*chans[0]*np.prod(s)*4)))
    # print(convert_byte(mod_m+inp_m + lab_m + cur_m + lab_m + argm_m))

    # print(convert_byte(cur_m*2 + mod_m + 2*lab_m)) ### + optimzer ~ 500MB


if __name__ == '__main__':
    main()













