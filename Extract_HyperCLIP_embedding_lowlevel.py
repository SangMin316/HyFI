import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from base.utils import instantiate_from_config, get_device 
import random
from tqdm import tqdm
import torch
from PIL import Image

from base.hycoclip.models import HyCoCLIP
from base.hycoclip.encoders.image_encoders import build_timm_vit
from base.hycoclip.encoders.text_encoders import TransformerTextEncoder
from base.hycoclip.tokenizer import Tokenizer 
from omegaconf import OmegaConf

import numpy as np
import cv2


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda')
print(device)



# You can select the hyperbolic model and download the check point 
# https://github.com/PalAvik/hycoclip


# ckpt_path = './base/hycoclip/checkpoints/hycoclip_vit_b.pth'
# ckpt_path = './base/hycoclip/checkpoints/meru_vit_s.pth'
# ckpt_path = './base/hycoclip/checkpoints/meru_vit_b.pth'
# ckpt_path = './base/hycoclip/checkpoints/meru_vit_l.pth'

brain_data = 'EEG'
for model_name in ['hycoclip_vit_s', 'hycoclip_vit_b', 'meru_vit_s', 'meru_vit_b', 'meru_vit_l']:

    if model_name in ['meru_vit_s', 'hycoclip_vit_s']:
        arch = 'vit_small_mocov3_patch16_224'
    elif model_name in ['meru_vit_l', 'hycoclip_vit_l']:
        arch = 'vit_large_patch16_224'
    elif model_name in ['meru_vit_b', 'hycoclip_vit_b']:
        arch = 'vit_base_patch16_224'



    visual = build_timm_vit(
        arch= arch,
        global_pool="token",
        use_sincos2d_pos=True
    )

    textual = TransformerTextEncoder(
        arch="L12_W512",     # L=12 layers, W=512 hidden dim, A=8 heads
        vocab_size=49408,       # BERT tokenizer vocab size
        context_length=77       #  
    )

    model = HyCoCLIP(
        visual=visual,
        textual=textual,
        embed_dim=512,           
        curv_init=1.0,
        learn_curv=True,
        entail_weight=0.2,
        use_boxes=True,
    )



    ckpt_path = f'./base/hycoclip/checkpoints/{model_name}.pth'

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint['model'] 
    model.load_state_dict(state_dict)


    model = model.to(device).eval()

    blur_mode = 'Gaussian'
    class ImageTextDataset(Dataset):
        def __init__(self, image_files, k_size):
            self.image_files = image_files
            self.k_size = k_size
            self.to_tensor = transforms.ToTensor()
        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            image = Image.open(self.image_files[idx]).convert("RGB")
            image = image.resize((224, 224))

            if blur_mode == 'Gaussian':
                image_np = np.array(image)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                blurred_np = cv2.GaussianBlur(image_np, (self.k_size, self.k_size), 0)
                image = Image.fromarray(cv2.cvtColor(blurred_np, cv2.COLOR_BGR2RGB))
            image = self.to_tensor(image)

            return image
    for k_size in [31]:
        original_model_name = model_name  

        for mode in ['train', 'test']:
            if mode == 'train':
                if brain_data == 'EEG':
                    data_path = "/your_data_path/training_images/data/training_images/"
                else:
                    data_path = "/your_data_path/Image_set/training_images/"
            else: 
                if brain_data == 'EEG':
                    data_path = "/your_data_path/training_images/data/test_images/"
                else:
                    data_path = "/your_data_path/Image_set/test_images/"

            image_files = glob.glob(data_path + "**/*.*", recursive=True)  
            image_files.sort()
   
            batch_size = 256
            dataset = ImageTextDataset(image_files,k_size)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

            model_name = original_model_name  



            print('model_name: ', model_name)
            image_latent_list = []

            with torch.no_grad():
                for images in tqdm(dataloader):
                    images = images.to(device)

                    image_features = model.encode_image(images, project=False)  # (B, D)
                    image_latent_list.append(image_features)
            image_latent = torch.cat(image_latent_list, dim=0)
            print('latent_features: ',image_latent.shape)

            if brain_data == 'EEG':
                torch.save(image_latent, f"/your_data_path/visual_feature/blur/{model_name}_gaussian_k_size{k_size}_{mode}.pt")
                print(f"Latent features saved! Shape: {image_latent.shape}")
            else:
                torch.save(image_latent, f"/your_data_path/visual_feature_MEG/blur/{model_name}_gaussian_k_size{k_size}_{mode}.pt")