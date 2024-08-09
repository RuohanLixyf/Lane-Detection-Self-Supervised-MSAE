from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn as nn
import config
import torchvision.transforms as transforms
import numpy as np
import config
from utils import PatchEmbed
from sklearn import preprocessing
import sys
import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block
from pos_embed import get_2d_sincos_pos_embed

def readTxt(file_path):
    img_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    return img_list
    
class RoadSequenceDatasetList(Dataset):

    def __init__(self, file_path, transforms):

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = []
        for i in range(5):
            data.append(torch.unsqueeze(self.transforms(Image.open(img_path_list[i])), dim=0))
        data = torch.cat(data, 0)
        label = Image.open(img_path_list[5]) ###Note this!!!!
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample
    
class RoadSequenceDatasetList_MASK(Dataset):# mask five pictures

    def __init__(self, file_path, transforms):

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size 
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h=int(128/16)
        w=int(256**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs
    
    def random_masking(self, x, mask_ratio):
            """
            Perform per-sample random masking by per-sample shuffling.
            Per-sample shuffling is done by argsort random noise.
            x: [N, L, D], sequence
            """#(128, 256, 3)
            x = torch.tensor(x)
            x = x.unsqueeze(dim=0)
            x = torch.einsum('nhwc->nchw', x)
            x1=x
            x=x.float()
            self.patch_embed = PatchEmbed((128, 256),16, 3, 768)
            # embed patches
            num_patches = self.patch_embed.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 768), requires_grad=False) 
            x = self.patch_embed(x)
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]
            N, L, D = x.shape  # batch, length, dim
            len_keep = int(L * (1 - mask_ratio))
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
            # visualize the mask
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
            mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
            mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
            x1 = torch.einsum('nchw->nhwc', x1)
            # masked image
            im_masked = x1 * (1 - mask)
           
            return im_masked, mask, ids_restore
    
    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = [] 
        for i in range(5):
                data1 = Image.open(img_path_list[i])
                data1.resize(( 128, 256))
                data1 = np.array(data1) / 255.
                data1.reshape == (128, 256, 3) 
                data1,mask,ids_restore=self.random_masking(data1,mask_ratio=config.mask_ratio)
                data.append(data1)
        data = torch.cat(data, 0)
        data=torch.squeeze(data,dim=0)
        data = torch.einsum('nchw->nwch', data)
        data = data.type(torch.cuda.FloatTensor)
        label = Image.open(img_path_list[4])     
        label = torch.squeeze(self.transforms(label))
        
        #show picture        
        # normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    # 归一化到[-1, 1]，公式是：(x-0.5)/0.5  
        # data0 = torch.squeeze(label)
        # data0 = torch.unsqueeze(data0, dim=0).cpu().numpy()
        # data0 = np.transpose(data0[-1], [1, 2, 0]) * 255      
        # data0 = Image.fromarray(data0.astype(np.uint8))
        # data0 = data0.convert("RGB")
        # data0.show()  
        
        sample = {'data': data, 'label': label}
        
        return sample


