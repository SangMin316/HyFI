import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base.data import load_data
from base.eeg_backbone import EEGProjectLayer
from base.hycoclip import lorentz as L

import torch
import argparse, os
from omegaconf import OmegaConf
import numpy as np
from collections import OrderedDict

import math
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
accelerator = "cuda"


# set the config 
config_path = '../configs/EEG.yaml'
config = OmegaConf.load(config_path)
config['data']['subjects'] = ['sub-04']
config['exp_setting'] = 'intra-subject'
config['brain_backbone'] = 'EEGProjectLayer'
config['vision_backbone'] = 'RN50'
config['c'] = 6



train_loader, val_loader, test_loader = load_data(config)


ckpt_path = 'HyFI/exp/intra-subject_HyFI_EEGProjectLayer_RN50/sub-04_seed42/checkpoints/last.ckpt'

z_dim =  1024 
c_num = 17
timesteps = [0, 250]
model = EEGProjectLayer(z_dim, c_num, timesteps)
model = model.to(device)


checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['state_dict']

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('brain.'):
        new_k = k[len('brain.'):]
    else:
        new_k = k
    new_state_dict[new_k] = v

model.load_state_dict(new_state_dict)


model.eval()


eeg_alpha = model.eeg_alpha.exp()
print('eeg_alpha:', eeg_alpha)


import glob
image_dir = "/your_data_path/test_images/"
image_paths = sorted(glob.glob(os.path.join(image_dir, "*", "*.jpg")))


ture_image_features = []

gallery_features = []
ground_truth_features = [] 
eeg_features = []

with torch.no_grad():
    for batch in test_loader:
        eeg = batch['eeg'].to(device)   # (B, C, T) or (B, S, C, T) depending on dataset
        img_features = batch['low_blur_feature'].to(device)
        blur_img_features = batch['hard_blur_feature'].to(device)
        path = batch['img_path']

        score = model.score(img_features)
        curv = model.curv.exp()

        eeg = model(eeg) 
        eeg = eeg*eeg_alpha
        eeg = L.exp_map0(eeg, curv)
        print('eeg: ',eeg.std())

        print('score mean:' ,score.mean())
        print('score std: ',score.std())

        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        img_features = model.image_projection1(img_features) 
        blur_img_features = model.image_projection2(blur_img_features)  
        
        print('blur_img_features: ',blur_img_features.std())

        blur_img_features = L.exp_map0(blur_img_features, curv)
        img_features = L.exp_map0(img_features, curv)

        interpolated_img_z = L.geodesic_interploation(img_features, blur_img_features,score, curv)
        
        gallery_features.append(interpolated_img_z)
        ground_truth_features.append(interpolated_img_z)
        ture_image_features.append(interpolated_img_z)
        eeg_features.append(eeg)

        break  

gallery_features = torch.cat(gallery_features, dim=0)         # image feature (N, D)
ground_truth_features = torch.cat(ground_truth_features, dim=0)  # GT image (N, D)
eeg_features = torch.cat(eeg_features, dim=0)                 # EEG query (N, D)
original_image = torch.cat(ture_image_features, dim=0)


dist_matrix = L.pairwise_dist(eeg_features, gallery_features, curv=curv).squeeze(0)    

top5_indices = torch.topk(dist_matrix, k=5, dim=1, largest=False).indices  # shape: (N, 5)

gt_indices = torch.arange(len(eeg_features), device=device)
gt_dists = L.pairwise_dist(eeg_features, gallery_features, curv=curv)
gt_dists =  gt_dists.diag()

sorted_indices = torch.argsort(gt_dists) 

n = 100  # number of samples

top_queries = sorted_indices[:n].tolist()
bottom_queries = sorted_indices[-n:].tolist()


def save_retrieval_results(query_idx_list, save_root, title):
    os.makedirs(save_root, exist_ok=True)
    
    for query_idx in query_idx_list:
        fig, axs = plt.subplots(1, 6, figsize=(18, 4))

        gt_img_path = image_paths[query_idx]
        gt_img = Image.open(gt_img_path).convert("RGB")
        axs[0].imshow(gt_img)

        axs[0].axis("off")
        name = path[query_idx].split('/')[-1]
        name = name.split('.')[0]

        for rank, img_idx in enumerate(top5_indices[query_idx]):
            img_path = image_paths[img_idx]
            img = Image.open(img_path).convert("RGB")
            axs[rank+1].imshow(img)
            axs[rank+1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_root, f"{title.lower()}_query_{name}.png"))
        plt.close()

save_retrieval_results(top_queries, "./retrieval_vis/top", "Top")
save_retrieval_results(bottom_queries, "./retrieval_vis/bottom", "Bottom")


gt_indices = torch.arange(len(eeg_features), device=device)
top1_preds = top5_indices[:, 0]
acc = (top1_preds == gt_indices).float().mean()
print(f"Top-1 accuracy: {acc:.2%}")