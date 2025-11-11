import torch,os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import logging
import open_clip
import gc
from tqdm import tqdm
import itertools


def load_data(config):
    exp_setting = config.get('exp_setting', 'intra-subject')
    
    if exp_setting == 'intra-subject':
        test_dataset = MEGDataset(config,mode='test')
        print('init test_dataset success')
        train_dataset = MEGDataset(config,mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False, drop_last=False,num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
        return train_loader, test_loader,test_loader
    
    elif exp_setting == 'inter-subject':
        subjects = config['data']['subjects']
        test_dataset = MEGDataset(config,mode='test')
        print('init test_dataset success')
        
        all_subjects = [f'sub-{i:02}' for i in range(1, 5)]
        leave_one_subjects = list(set(all_subjects) - set(subjects))
        leave_one_subjects_config = config
        leave_one_subjects_config['data']['subjects'] = leave_one_subjects
        val_dataset = MEGDataset(leave_one_subjects_config,mode='test')
        print('init val_dataset success')
        train_dataset = MEGDataset(leave_one_subjects_config,mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False, drop_last=False,num_workers=25)#, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['data']['val_batch_size'], shuffle=False, drop_last=False,num_workers=32)#, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True, drop_last=False, num_workers=32)#, pin_memory=True)
        return train_loader, val_loader, test_loader
    
class MEGDataset(Dataset):
    def __init__(self, config, mode):
        self.config= config
        self.data_dir = '/your_data_path/things_meg'

        self.subjects = config['data']['subjects']
        print(f'subjects:{self.subjects}')
        self.mode = mode
        self.name = config['name']
        self.model_type = config['data']['model_type']
        self.selected_ch = config['data']['selected_ch']
        self.channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                        'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                        'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                        'O1', 'Oz', 'O2']
        if self.selected_ch == "None":
            self.selected_ch = self.channels
    
        self.avg = config['data'][f"{mode}_avg"]

        self.blur_type = config['data']['blur_type']

        self.timesteps = config['data']['timesteps']

        self.n_cls = 1654 if self.mode=='train' else 200
        self.per_trials = 1 if self.mode=='train' else 12

        self.data_paths = [os.path.join(self.data_dir,subject,f'{mode}.pt') for subject in self.subjects]
        self.loaded_data= [self.load_data(data_path) for data_path in self.data_paths]
        
        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject*len(self.subjects)

        vlm_name = self.config['data']['model_type']
        if vlm_name == 'ViT-B-16':
            vlm_name = 'ViT_B_16'
        elif vlm_name == 'ViT-B-32':
            vlm_name = 'ViT_B_32'
        elif vlm_name == 'ViT-L-14':
            vlm_name = 'ViT_L_14'
        elif vlm_name == 'ViT-H-14':
            vlm_name = 'ViT_H_14'
            
        soft_k = self.config['Blur']['soft_k']
        soft_g = self.config['Blur']['soft_g']
        hard_k = self.config['Blur']['hard_k']


        feature_dir = '/your_data_path/visual_feature_MEG/blur'
        low_blur_filename = f"{vlm_name}_k{soft_k}_g{soft_g}_{mode}.pt"
        low_blur_path = os.path.join(feature_dir, low_blur_filename)
        
        hard_blur_filename = f"{vlm_name}_gaussian_k_size{hard_k}_{mode}.pt"
        hard_blur_path = os.path.join(feature_dir, hard_blur_filename)

        self.low_blur_feature = torch.load(low_blur_path, map_location="cpu")
        self.hard_blur_feature = torch.load(hard_blur_path, map_location="cpu")


    def load_data(self,data_path):
        logging.info(f"----load {data_path.rsplit('1000HZ',1)[-1]}----")
        loaded_data = torch.load(data_path, weights_only=False)
        loaded_data['eeg']=torch.from_numpy(loaded_data['eeg'])
        
        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:,:,selected_idx]
        if self.avg:
            avg_data={}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['img'] = np.array(loaded_data['img'])#[:,0]

            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1,*loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['img'] = loaded_data['img'].reshape(-1)

            loaded_data = _data
        
        
        for k,v in loaded_data.items():
            if k in ['eeg','label','img','text','session']:
                logging.info(f"{k}: {v.shape}")
        return loaded_data    
    
    
    def __getitem__(self, index):
        
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.avg:
            eeg_mean = eeg
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index//self.per_trials].float()

        img_path = self.loaded_data[subject]['img'][trial_index]

        
        hard_blur_feature = self.hard_blur_feature[trial_index]
        low_blur_feature = self.low_blur_feature[trial_index]
        
        sample  = {
            'idx': index,
            'eeg': eeg[:,self.timesteps[0]:self.timesteps[1]],
            'img_path': img_path,
            'low_blur_feature': low_blur_feature,
            'hard_blur_feature': hard_blur_feature,
            'subject': subject,
            'eeg_mean': eeg_mean[:,self.timesteps[0]:self.timesteps[1]],
        }
        return sample
    
    def __len__(self):
        return self.trial_all_subjects
    