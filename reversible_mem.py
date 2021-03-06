import torch
from models.revunet_3D import RevUnet3D, Interpolate
import numpy as np

import models



def get_mod_details(model):
    ret = {'name' : [], 'layer' : []}
    for name, layer in model.named_modules():
        #print(name, layer, type(layer))
        if name != "" and (isinstance(layer, models.revunet_3D.softmax) or isinstance(layer, torch.nn.modules.upsampling.Upsample) or isinstance(layer, torch.nn.Conv3d) or isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.GroupNorm) or isinstance(layer, torch.nn.MaxPool3d)):
            ret['name'].append(name)
            # print(name)
            ret['layer'].append(layer)
    return ret






def get_activations_shapes(layers, x):
    no_rev_shapes = [x.shape]
    with torch.no_grad():
        for n, l in zip(layers['name'], layers['layer']):
            print(n)
            if 'reversible' not in n:
                x = l(x)
                shapes.append(x.shape)
                
    return shapes

def get_activations_shapes_as_dict(layers, x):
    shapes = {'input':x.shape}
    with torch.no_grad():
        for n, l in zip(layers['name'], layers['layer']):
            if 'reversible' not in n:
                x = l(x)
                shapes[n] = x.shape
            else:
                x1, x2 = torch.split(x, x.shape[1]//2, dim = 1)
                y1 = l(x1) + x2
                shapes[n] = y1.shape
    return shapes





def memory_cosumption(shapes, float_types='float'):
    m = 0
    all_types = {'float':4, 'double':8, 'half':2}
    for s in shapes:
        m += np.prod(s)*all_types[float_types]
        print(s, np.prod(s)*all_types[float_types])
    return m

def forward_memory_cosumption_with_peak(shapes, float_types='float'):
    cur_m = 0
    max_m = 0
    all_types = {'float':4, 'double':8, 'half':2}
    
    enc, dec = -1, -1
    ind = 0
    for k in list(shapes.keys()):
        s = shapes[k]
        txt = "" + k
        
        if 'reversible' not in k:
            cur_m += np.prod(s)*all_types[float_types]
            max_m += np.prod(s)*all_types[float_types]
            txt += " :: saved !"
        elif ('encoders' in k) or ('decoders' in k):
            tmp_enc = int(k[9])
            if tmp_enc == enc:
                tmp_max_m += np.prod(s)*all_types[float_types]*2
                if max_m < tmp_max_m:
                    max_m = tmp_max_m
            else:
                enc = tmp_enc
                tmp_max_m = cur_m + np.prod(s)*all_types[float_types]
                if max_m < tmp_max_m:
                    max_m = tmp_max_m
            txt += " :: notsa !"

            if (ind < len(list(shapes.keys()))-1) and ('encoders' in k):
                a = list(shapes.keys())[ind][9]
                b = list(shapes.keys())[ind+1][9]
                # print(a, b)
                # a = int(a) 
                # b = int(b)

                if a < b:
                    cur_m += np.prod(s)*all_types[float_types]
                    max_m += np.prod(s)*all_types[float_types]

        ind += 1
        print(txt)
        print(convert_byte(max_m))
            
                
        
        #print(s, np.prod(s)*all_types[float_types])
    return cur_m, max_m


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
    inchan = 1
    chanscale = 1
    chans = [i//chanscale for i in [64, 128, 256, 512, 1024]]
    outsize = 14
    interp = None
    mod = RevUnet3D(inchan, chans, outsize, interp)

    layers = get_mod_details(mod)

    fact = 1
    # s = (80, 80, 32)
    # s = (112,112,48)
    s = (256,256,112)
    # x = torch.from_numpy(np.random.rand(1,1,int(round(512*fact)),int(round(512*fact)),int(round(200*fact)))).float()
    # y = torch.from_numpy(np.random.rand(1,outsize,int(round(512*fact)),int(round(512*fact)),int(round(200*fact)))).float()
    # argmax = torch.from_numpy(np.random.rand(1,outsize,int(round(512*fact)),int(round(512*fact)),int(round(200*fact)))).float()
    x = torch.from_numpy(np.random.rand(1,1,s[0], s[1], s[2])).float()
    y = torch.from_numpy(np.random.rand(1,outsize,s[0], s[1], s[2])).float()
    # argmax = torch.from_numpy(np.random.rand(1,outsize,s[0], s[1], s[2])).float()

    acts = get_activations_shapes_as_dict(layers, x)


    cur_m, max_m = forward_memory_cosumption_with_peak(acts)

    mod_m = model_memory(mod)
    lab_m = labels_mem(y)
    inp_m = labels_mem(x)
    bac_m = chans[0]*np.prod(x.shape)*4
    # argm_m = validat_arg_memory(argmax)

    # print(convert_byte(cur_m*2 + mod_m + lab_m))
    # print(convert_byte(max_m + mod_m + lab_m))
    print('model :', convert_byte(mod_m))
    print('input :', convert_byte(mod_m+inp_m))
    print('label :', convert_byte(mod_m+inp_m + lab_m))
    print('forwa :', convert_byte(mod_m+inp_m + lab_m + max_m))
    print('backw :', convert_byte(mod_m+inp_m + lab_m + max_m + bac_m))

    # print('backw :', convert_byte(mod_m+inp_m + lab_m + cur_m + back_m))

    # print('lab', convert_byte(mod_m + + lab_m))
    # print('for', convert_byte(mod_m + lab_m + cur_m))
    # print('bac', convert_byte(mod_m+ 2*lab_m + cur_m))
    # print('arg', convert_byte(mod_m+ 2*lab_m + cur_m + argm_m))


if __name__ == '__main__':
    main()













