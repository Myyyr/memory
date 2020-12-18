import numpy as np 
import torch
# from models.unet_3D import unet_3D
from models.revunet_3D import revunet_3D
from models.memoryUtils import Hook



import models.memoryUtils as atlasUtils

import json
import os



def maxmem():
	return torch.cuda.max_memory_allocated()

def curmem():
	return torch.cuda.memory_allocated()


def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.2f %s" % (size, x)
        size /= 1024.0

    return size



def apply_hook(net):
	hookF = []
	hookB = []
	for i,layer in enumerate(list(net.children())):
	    if not isinstance(layer,torch.nn.ReLU) and not isinstance(layer,torch.nn.LeakyReLU):
	        print('Hooked to {}'.format(layer))
	        hookF.append(Hook(layer))
	        hookB.append(Hook(layer,backward=True))

	return hookF, hookB



def main():
	gpu = '2'
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu
	device = torch.device("cuda")
	memory_callback = {}

	inchan = 1
	chanscale = 2
	chans = [i//chanscale for i in [64, 128, 256, 512, 1024]]
	outsize = 14
	interp = (512,512,198)


	# mod = unet_3D(chans, n_classes=outsize, in_channels=inchan, interpolation = interp)
	mod = RevUnet3D(inchan, chans, outsize, interp)
	mod.to(device)
	hookF, hookB = apply_hook(mod)
	memory_callback['model'] = {'max' : maxmem(), 'cur' : curmem()}

	fact = 0.1

	x = torch.from_numpy(np.random.rand(1,1,int(round(512*fact)),int(round(512*fact)),int(round(198*fact)))).float()
	x = x.to(device)
	memory_callback['input'] = {'max' : maxmem(), 'cur' : curmem()}
	y = torch.from_numpy(np.random.rand(1,outsize,512,512,198)).float()
	y = y.to(device)
	memory_callback['output'] = {'max' : maxmem(), 'cur' : curmem()}

	opti = torch.optim.SGD(mod.parameters(), lr=0.01)
	opti.zero_grad()

	def loss(outputs, labels):
	    return atlasUtils.atlasDiceLoss(outputs, labels)
	loss = loss


	mod.train()
	out = mod(x)
	del x
	memory_callback['forward'] = {'max' : maxmem(), 'cur' : curmem()}

	l = loss(out, y)
	del y, out
	memory_callback['loss'] = {'max' : maxmem(), 'cur' : curmem()}

	l.backward()
	memory_callback['backward'] = {'max' : maxmem(), 'cur' : curmem()}

	opti.step()
	memory_callback['step'] = {'max' : maxmem(), 'cur' : curmem()}

	memory_callback['hookF'] = []
	memory_callback['hookB'] = []

	for i,j in zip(hookF, hookB):
		memory_callback['hookF'].append({'max' : i.max_mem, 'cur' : i.cur_mem})
		memory_callback['hookB'].append({'max' : j.max_mem, 'cur' : j.cur_mem})

	# with open('callback_memory.json', 'w') as f:
	# 	json.dump(memory_callback, f, indent=4)

	with open('callback_rev_memory.json', 'w') as f:
		json.dump(memory_callback, f, indent=4)


if __name__ == '__main__':
	main()






