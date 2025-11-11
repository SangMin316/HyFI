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



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
accelerator = "cuda"

config_path = '../configs/EEG.yaml'
config = OmegaConf.load(config_path)
config['data']['subjects'] = ['sub-08']
config['exp_setting'] = 'intra-subject'
config['brain_backbone'] = 'EEGProjectLayer'
config['vision_backbone'] = 'RN50'
config['c'] = 6



train_loader, val_loader, test_loader = load_data(config)


ckpt_path = 'HyFI/exp/intra-subject_HyFI_EEGProjectLayer_RN50/sub-08_seed42/checkpoints/last.ckpt'


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

curv = model.curv.exp()
print(curv)

model.eval()

def distance_from_root(x):
    x_time = torch.sqrt(1/curv + torch.sum(x**2, dim=-1, keepdim=True)) # calculate the time component
    val = (curv * x_time).clamp(min=1.0 + 1e-5)
    return (1 / curv**0.5) * torch.acosh(val).unsqueeze(-1) # distance from the root


eeg_alpha = model.eeg_alpha.exp()
print('eeg_alpha:', eeg_alpha)

all_interpolated_z = []
all_eeg = []
all_img_z = []
all_blur_z = []


with torch.no_grad():
    for batch in test_loader:
        eeg = batch['eeg'].to(device)   # (B, C, T) or (B, S, C, T) depending on dataset
        img_features = batch['low_blur_feature'].to(device)
        blur_img_features = batch['hard_blur_feature'].to(device)


        score = model.score(img_features)
        eeg = model(eeg) 
        eeg = eeg*eeg_alpha
        eeg = L.exp_map0(eeg, curv)
      
        img_features = img_features/img_features.norm(dim=-1, keepdim=True)

        img_features = model.image_projection1(img_features)  
        blur_img_features = model.image_projection2(blur_img_features)  

        blur_img_features = L.exp_map0(blur_img_features, curv)
        img_features = L.exp_map0(img_features, curv)

        interpolated_img_z = L.geodesic_interploation(img_features, blur_img_features,score, curv)

        all_interpolated_z.append(interpolated_img_z)
        all_eeg.append(eeg)
        all_img_z.append(img_features)
        all_blur_z.append(blur_img_features)


all_eeg = torch.cat(all_eeg, dim=0)
all_img_z = torch.cat(all_img_z, dim=0)
all_blur_z = torch.cat(all_blur_z, dim=0)
all_interpolated_z = torch.cat(all_interpolated_z, dim=0)

print(all_eeg.shape)
eeg = distance_from_root(all_eeg)
blur_img_features = distance_from_root(all_blur_z)
img_features = distance_from_root(all_img_z)
interpolated_img_z = distance_from_root(all_interpolated_z)

eeg_features = eeg.to("cpu").detach()
blur_img_features = blur_img_features.to("cpu").detach()
img_features = img_features.to("cpu").detach()
interpolated_img_z = interpolated_img_z.to("cpu").detach()




plt.figure(figsize=(5,2.5))
sns.histplot(data=eeg_features.squeeze(), bins='auto', stat="percent", kde=True, element="step", alpha=0.5, label='EEG')
sns.histplot(data=blur_img_features.squeeze(), bins='auto', stat="percent", kde=True, element="step", alpha=0.5, label='L')
sns.histplot(data=interpolated_img_z.squeeze(), bins='auto', stat="percent", kde=True, element="step", alpha=0.5, label='I')
sns.histplot(data=img_features.squeeze(), bins='auto', stat="percent", kde=True, element="step", alpha=0.5, label='H')

plt.xlim(0.45, 1.2)
plt.xlabel((r'$\Vert \mathbf{\tilde{p}} \Vert$'))
plt.ylabel('% of samples')


save_dir = "/HyFI/Analysis/feature_distribution_from_the_root.png"


plt.savefig(save_dir, bbox_inches="tight", dpi=300)

