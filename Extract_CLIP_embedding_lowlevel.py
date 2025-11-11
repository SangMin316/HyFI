import torch
from torchvision import transforms
# from diffusers import AutoencoderKL
from PIL import Image
import os
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
import random
from tqdm import tqdm


import open_clip
import clip
import torch
from PIL import Image
import numpy as np
import cv2


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device('cuda')
print(device)

brain_data = 'EEG'

model_list = ['RN101','ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-H-14']
for model_name in model_list:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name in ['RN50', 'RN101','ViT-B/32', 'ViT-B/16', 'ViT-L/14']:
        model, preprocess = clip.load(model_name, device=device)

    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained='laion2b_s32b_b79k',
            device=device
        )


    model = model.to(device)

    blur_mode = 'Gaussian'
    # blur_mode = 'Noise'
    # blur_mode = 'LowRes'



    class ImageTextDataset(Dataset):
        def __init__(self, image_files, k_size, preprocess):
            self.image_files = image_files
            self.preprocess = preprocess
            self.k_size = k_size

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
            
            elif blur_mode == 'Noise':
                image_np = np.array(image).astype(np.float32) / 255.0
                noise = np.random.normal(0, 0.3, image_np.shape)
                noisy_np = np.clip(image_np + noise, 0, 1)
                noisy_np = (noisy_np * 255).astype(np.uint8)
                image = Image.fromarray(noisy_np)

            elif blur_mode == 'LowRes':
                # downsample → upsample
                low_res_size = (32, 32)
                image = image.resize(low_res_size, resample=Image.BICUBIC)
                image = image.resize((224, 224), resample=Image.BICUBIC)
            image = self.preprocess(image)

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

            model_name = original_model_name 

            batch_size = 256
            dataset = ImageTextDataset(image_files,k_size, preprocess)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            
            if model_name == "ViT-B/32":
                model_name = "ViT_B_32"
            elif model_name == "ViT-B/16":
                model_name = "ViT_B_16"
            elif model_name == "ViT-L/14":
                model_name = "ViT_L_14"
            elif model_name == 'ViT-H-14':
                model_name = "ViT_H_14"

            print('model_name: ', model_name)
            image_latent_list = []
            with torch.no_grad():
                for images in tqdm(dataloader):
                    images = images.to(device)

                    image_features = model.encode_image(images)
                    image_latent_list.append(image_features)
    
            image_latent = torch.cat(image_latent_list, dim=0)
            print('latent_features: ',image_latent.shape)

            if brain_data == 'EEG':
                torch.save(image_latent, f"/your_data_path/visual_feature/blur/{model_name}_gaussian_k_size{k_size}_{mode}.pt")
                print(f"Latent features saved! Shape: {image_latent.shape}")
            else:
                torch.save(image_latent, f"/your_data_path/visual_feature_MEG/blur/{model_name}_gaussian_k_size{k_size}_{mode}.pt")