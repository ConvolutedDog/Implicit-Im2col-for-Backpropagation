import os
import torch

def save_tensor(tensor_x, layer_name):
    pth_dir = "DIRECTORYOFTMPFILE"
    files = os.listdir(pth_dir)
    for file in files:
        if not '.pth' in file:
            files.remove(file)
    if len(files) == 0:
        pth_name = '0.pth'
    else:
        max_num_of_pth = max([int(x.split('.')[0]) for x in files])
        pth_name = str(max_num_of_pth+1)+'.pth'
    print(layer_name+' save '+pth_name)
    torch.save(tensor_x, os.path.join(pth_dir, pth_name))