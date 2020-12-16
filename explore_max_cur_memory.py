import torch
import numpy as np



def convert_byte(v):
    units = {'Bytes':1,'KB':1e-3, 'MB':1e-6, 'GB':1e-9}
    tmp = 'Bytes'
    for k in list(units.keys()):
        if int(v*units[k]) == 0:
            return v*units[tmp], tmp
        tmp = k
    return v*units[tmp], tmp


def prt_mem(txt, d, flt_type = 4):
    a = torch.cuda.max_memory_allocated()
    b = torch.cuda.memory_allocated()
    print(txt,':' ,convert_byte(a) , convert_byte(b), ((a-b)/(d*flt_type)))
    return a,b

def explore_conv(x, l):
    prt_mem('start', np.prod(x.shape[2:]))
    x = x.cuda()
    prt_mem('x', np.prod(x.shape[2:]))
    l = l.cuda()
    prt_mem('l', np.prod(x.shape[2:]))
    y = l(x)
    prt_mem('forward', np.prod(x.shape[2:]))
    del x, l, y

def reset():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def main():
    im_shape = [1,2,256,256,200]

    for i in range(1,10):
        tmp_shape =  im_shape[:2] + [j//i for j in im_shape[2:]]
        print('\n\n\n###### ',tmp_shape,' ######')

        x = torch.from_numpy(np.random.rand(*tmp_shape)).float()

        l = torch.nn.Conv3d(2,2,3,padding=1,bias=False)

        explore_conv(x,l)

        reset()




if __name__ == '__main__':
    main()