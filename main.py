import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
from torch.optim import AdamW, Adam, SGD
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from collections import Counter
from scipy.stats import norm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from base.utils import *
from base.hycoclip import lorentz as L

torch.cuda.empty_cache()


def load_model(config,train_loader,test_loader):
    model = {}
    for k,v in config['models'].items():
        print(f"init {k}")
        model[k] = instantiate_from_config(v)

    pl_model = PLModel(model,config,train_loader,test_loader)
    return pl_model

class PLModel(pl.LightningModule):
    def __init__(self, model,config,train_loader,test_loader, model_type = 'RN50'):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)

        self.criterion = Hyperbolic_CL()

        self.all_predicted_classes = []
        self.all_true_labels = []

        self.z_dim = self.config['z_dim']
        print(self.z_dim)
        self.sim = np.ones(len(train_loader.dataset))

        self.mAP_total = 0
        self.match_similarities = []

        self.visual_alpha = 1
    
    
    def forward(self, batch,sample_posterior=False):
 
        eeg = batch['eeg']
        img_z_semantic  = batch['low_blur_feature']
        img_z_perceptual = batch['hard_blur_feature']

        eeg_alpha = self.brain.eeg_alpha.exp()
        visual_alpha = self.visual_alpha
        curv = self.brain.curv.exp()

        eeg_z = self.brain(eeg)
        #calculate the interpolation coefficient
        score = self.brain.score(img_z_semantic)

        #We find the that normalized the sementic features are more effective
        img_z_semantic = img_z_semantic / img_z_semantic.norm(dim=-1, keepdim=True)

        # map to Lorentz space
        eeg_z = eeg_alpha*eeg_z
        eeg_z = L.exp_map0(eeg_z, curv)

        # apply linear layer
        img_z_semantic = self.brain.image_projection1(img_z_semantic)
        img_z_perceptual = self.brain.image_projection2(img_z_perceptual)

        # map to Lorentz space
        img_z_semantic = visual_alpha*img_z_semantic
        img_z_semantic = L.exp_map0(img_z_semantic, curv)
        img_z_perceptual = visual_alpha*img_z_perceptual
        img_z_perceptual = L.exp_map0(img_z_perceptual, curv)


        # interpolation following the geodesic
        interpolated_z = L.geodesic_interploation(img_z_semantic, img_z_perceptual,score,curv)

        # apply contrastive learning
        loss = self.criterion(eeg_z, interpolated_z, curv)

        return eeg_z, interpolated_z, loss, curv

    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss, curv = self(batch,sample_posterior=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        
        similarity = L.pairwise_dist(eeg_z, img_z, curv)
        top_kvalues, top_k_indices = similarity.topk(5,largest=False, dim=-1)


        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_predictions = all_predicted_classes[:, 0]
            top_1_correct = top_1_predictions == all_true_labels
            top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
            self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []

        return loss


    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss, curv = self(batch,sample_posterior=True)

        B, D = eeg_z.shape
        label = torch.arange(0, batch_size).to(self.device)

        similarity = L.pairwise_dist(eeg_z, img_z,curv)
        top_kvalues, top_k_indices = similarity.topk(5,largest=False, dim=-1)

        self.all_predicted_classes.append(top_k_indices.cpu().numpy())

        self.all_true_labels.extend(label.cpu().numpy())

        self.log('val_loss', loss, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        
        return loss
    

    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []


    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss, curv = self(batch,sample_posterior=True)
      
        low_blur = batch['low_blur_feature']
        score = self.brain.score(low_blur)
        self.score = score.mean()


        self.log('test_loss', loss, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        label = torch.arange(0, batch_size).to(self.device)

        similarity = L.pairwise_dist(eeg_z, img_z,curv)
        top_kvalues, top_k_indices = similarity.topk(5,largest=False, dim=-1)

        mAP = 0.0

        # label = batch['label']
        self.all_true_labels.extend(label.cpu().numpy())
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())
        self.mAP_total += mAP

        for i in range(similarity.shape[0]):
            true_index = i
            sims = similarity[i, :]
            sorted_indices = torch.argsort(sims)
            rank = (sorted_indices == true_index).nonzero()[0][0] + 1
            ap = 1 / rank
            self.mAP_total += ap
        
        return loss


    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)

        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)

        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0

        
        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)
        self.log('score', self.score, sync_dist=True)


        self.all_predicted_classes = []
        self.all_true_labels = []

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return  {'test_loss': avg_test_loss.item(), 'test_top1_acc': top_1_accuracy.item(),'test_top5_acc':top_k_accuracy.item(),'mAP':self.mAP,'similarity':self.match_similarities, 'score': self.score}
        
    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr = self.config['train']['lr'], weight_decay= 1e-4)

        return [optimizer]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/EEG.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--exp_setting",
        type=str,
        # default='inter-subject',
        default='intra-subject',
        help="the exp_setting",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="train epoch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default = 3e-4,
        # default = 3e-5, for inter setting
        help="lr",
    )
    parser.add_argument(
        "--brain_backbone",
        type=str,
        default = "EEGProjectLayer",
        help="brain_backbone",
    )

    parser.add_argument(
        "--vision_backbone",
        type=str,
        default = "RN50",
        help="vision_backbone",
    )

    parser.add_argument(
        "--brain_data",
        type=str,
        default = "EEG",
        help="data",
    )
    
    parser.add_argument(
        "--c",
        type=int,
        default=6,
        help="c",
    )
    parser.add_argument('--gpu', type=str, default="1", help='gpu id')



    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    accelerator = "cuda"
    top1_acc_list = []
    top5_acc_list = []
    mac_list = []
    score_list = []
    
    if config['data']['selected_ch'] == 'ALL':
        config['c_num'] = 63

    elif config['data']['selected_ch'] == 'O':
        config['c_num'] = 8

    elif config['data']['selected_ch'] =='P':
        config['c_num'] = 9

    elif config['data']['selected_ch'] == 'C':
        config['c_num'] = 14

    elif config['data']['selected_ch'] =='T':
        config['c_num'] = 10

    elif config['data']['selected_ch'] == 'F':
        config['c_num'] = 22

    elif config['data']['selected_ch'] =='OP':
        config['c_num'] = 17

    elif config['data']['selected_ch'] =='OPT':
        config['c_num'] = 27

    elif config['data']['selected_ch'] == 'OPCT':
        config['c_num'] = 41
    


    ##import user lib
    if opt.brain_data == "EEG":
        from base.data import load_data
        sub_list = range(1,11)
    else:
        from base.data_meg import load_data
        sub_list = range(1,5)

    for subject_idx in sub_list:
    # for subject_idx in [4]:
        subject_idx = f"sub-{str(subject_idx).zfill(2)}"
        print('subject_idx: ', subject_idx)
        config['data']['subjects'] = [subject_idx]


        pretrain_map = {
            'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
            'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
            'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 512},
            'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224), 'z_dim': 512},
            'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
            'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
            'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
            'meru_vit_s': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 512},
            'meru_vit_b': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 512},
            'meru_vit_l': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 512},
            'hycoclip_vit_s': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 512},
            'hycoclip_vit_b': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 512},
            'hycoclip_vit_l': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 512}
        }


        config['z_dim'] = pretrain_map[opt.vision_backbone]['z_dim']
        print(config)

        os.makedirs(config['save_dir'],exist_ok=True)
        logger = TensorBoardLogger(config['save_dir'], name=config['name'], version=f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}")
        os.makedirs(logger.log_dir,exist_ok=True)
        shutil.copy(opt.config, os.path.join(logger.log_dir,opt.config.rsplit('/',1)[-1]))

        train_loader, val_loader, test_loader = load_data(config)

        print(f"train num: {len(train_loader.dataset)},val num: {len(val_loader.dataset)}, test num: {len(test_loader.dataset)}")
        pl_model = load_model(config, train_loader, test_loader)

        checkpoint_callback = ModelCheckpoint(save_last=True)

        if config['exp_setting'] == 'inter-subject':
            early_stop_callback = EarlyStopping(
                monitor='val_top1_acc',
                # monitor='train_loss',
                min_delta=0.001,     
                patience=5, 
                verbose=False,
                mode='max' 
                # mode='min' 
            )
        else:
            early_stop_callback = EarlyStopping(
                monitor='train_loss',
                min_delta=0.001,
                patience=5,
                verbose=False,  
                mode='min' 
            )

        trainer = Trainer(
            log_every_n_steps=10,
            callbacks=[early_stop_callback, checkpoint_callback],
            max_epochs=config['train']['epoch'],
            devices=1,
            accelerator=accelerator,
            logger=logger
        )
        print(trainer.logger.log_dir)

        # ckpt_path = 'last' #None
        ckpt_path = None #None

        trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader,ckpt_path=ckpt_path)

        if config['exp_setting'] == 'inter-subject':
            test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
        else:
            test_results = trainer.test(ckpt_path='last', dataloaders=test_loader)

        with open(os.path.join(logger.log_dir,'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)

        top1_acc_list.append(test_results[0]['test_top1_acc'])  
        top5_acc_list.append(test_results[0]['test_top5_acc'])
        mac_list.append(test_results[0]['mAP'])
        score_list.append(test_results[0]['score'])

        
    print('top1_acc_list:', np.array(top1_acc_list).mean())
    print(top1_acc_list)
    print('top5_acc_list:', np.array(top5_acc_list).mean())
    print(top5_acc_list)
    print('mac_list:', np.array(mac_list).mean())
    print(mac_list)
    print('score_list:', np.array(score_list).mean())
    print(score_list)


if __name__=="__main__":
    main()
