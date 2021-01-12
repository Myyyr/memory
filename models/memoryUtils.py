import torch
import torch.nn as nn



class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

        self.max_mem = []
        self.cur_mem = []
        self.name = []

    def hook_fn(self, module, input, output):
        
        self.max_mem.append(torch.cuda.max_memory_allocated())
        self.cur_mem.append(torch.cuda.memory_allocated())
        self.name.append(module.name)
        print('Hook is called on {}, max_mem : {}, cur_mem : {}'.format(module, convert_bytes(self.max_mem[-1]), convert_bytes(self.cur_mem[-1])))



def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.2f %s" % (size, x)
        size /= 1024.0

    return size



def softDice(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    if nonSquared:
        union = (pred).sum() + (target).sum()
    else:
        union = (pred * pred).sum(dim=(1, 2, 3)) + (target * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + smoothing) / (union + smoothing)

    dice[dice != dice] = dice.new_tensor([1.0])

    return dice.mean()

def dice(pred, target):
    predBin = (pred > 0.5).float()
    return softDice(predBin, target, 0, True).item()

def diceLoss(pred, target, nonSquared=False):
    return 1 - softDice(pred, target, nonSquared=nonSquared)

def atlasDiceLoss(outputs, labels, nonSquared=False):
    n_classe = 14
    chunk = list(outputs.chunk(n_classe, dim=1))
    
    s = chunk[0].shape
    

    for i in range(n_classe):
        chunk[i] = chunk[i].view(s[0], s[2], s[3], s[4])
        


    chunkMask = list(labels.chunk(n_classe, dim=1))
    s = chunkMask[0].shape
    
    for i in range(n_classe):
        chunkMask[i] = chunkMask[i].view(s[0], s[2], s[3], s[4])


    losses = []
    for i in range(n_classe):
        losses.append(diceLoss(chunk[i], chunkMask[i], nonSquared=nonSquared))

    # print('###END LOSS###')

    return sum(losses) / n_classe


