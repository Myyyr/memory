import numpy as np 
import torch
from models.unet_3D import unet_3D
from models.memoryUtils import Hook

def apply_hook(net):
	hookF = []
	hookB = []
	for i,layer in enumerate(list(net.children())):
	    if not isinstance(layer,nn.ReLU) and not isinstance(layer,nn.LeakyReLU):
	        print('Hooked to {}'.format(layer))
	        hookF.append(Hook(layer))
	        hookB.append(Hook(layer,backward=True))

	return hookF, hookB



def main():
	memory_callback = {}

	inchan = 1
	chanscale = 2
	chans = [i//chanscale for i in [64, 128, 256, 512, 1024]]
	outsize = 14
	interp = (512,512,198)
	mod = unet_3D(chans, n_classes=outsize, in_channels=inchan, interpolation = interp)

	layers = get_mod_details(mod)

	fact = 0.1

	x = torch.from_numpy(np.random.rand(1,1,int(round(512*fact)),int(round(512*fact)),int(round(198*fact)))).float()
	y = torch.from_numpy(np.random.rand(1,outsize,512,512,198)).float()

	



if __name__ == '__main__':
	main()






