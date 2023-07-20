import os
import glob
import torch
import shutil
import numpy as np
from torch import nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import NeuralPhysDataset, NeuralPhysRefineDataset
from model_utils import (EncoderDecoder64x1x1,
                         RefineSinglePendulumModel,
                         RefineDoublePendulumModel,
                         RefineElasticPendulumModel)

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

class VisDynamicsModel(pl.LightningModule):

    def __init__(self,
                 lda: 1.0,
                 beta: 1.0,
                 lr: float=1e-4,
                 seed: int=1,
                 if_cuda: bool=True,
                 if_test: bool=False,
                 gamma: float=0.5,
                 log_dir: str='logs',
                 train_batch: int=512,
                 val_batch: int=256,
                 test_batch: int=256,
                 num_workers: int=8,
                 model_name: str='encoder-decoder-64',
                 data_filepath: str='data',
                 dataset: str='single_pendulum',
                 num_frames: int=60,
                 lr_schedule: list=[20, 50, 100]) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.global_iter = 0
        assert self.hparams.if_cuda
        self.kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True}
        # create visualization saving folder if testing
        self.pred_log_dir = os.path.join(self.hparams.log_dir, 'predictions')
        self.var_log_dir = os.path.join(self.hparams.log_dir, 'variables')
        if not self.hparams.if_test:
            mkdir(self.pred_log_dir)
            mkdir(self.var_log_dir)
        self._build_model()

    def _build_model(self):
        if self.hparams.model_name == 'encoder-decoder-64':
            self.model = EncoderDecoder64x1x1(in_channels=3)
        if self.hparams.model_name == 'refine-64':
            if self.hparams.dataset == 'single_pendulum':
                self.model = RefineSinglePendulumModel(in_channels=64)
            if self.hparams.dataset == 'double_pendulum':
                self.model = RefineDoublePendulumModel(in_channels=64)
            if self.hparams.dataset == 'elastic_pendulum':
                self.model = RefineElasticPendulumModel(in_channels=64)
            if self.hparams.if_test:
                self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)

    def encoder_decoder_loss(self, output, target, mu, logvar):
        mse_loss = torch.nn.MSELoss(reduction='sum')
        rec_loss = mse_loss(output, target)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        rec_loss = rec_loss / target.shape[0]
        kl_diverge = kl_diverge / target.shape[0]
        reg_loss = self.reg_loss_(mu)

        return (
            (rec_loss + self.hparams.lda * reg_loss + self.hparams.beta * torch.abs(kl_diverge)),
            rec_loss, 
            reg_loss,
            kl_diverge,   
        )
    
    def refine_loss(self, output, latent, mu, logvar):
        rec_loss = 0.5 * (logvar + (output - mu) ** 2 / logvar.exp())
        rec_loss = torch.sum(rec_loss) / mu.shape[0]
        reg_loss = self.reg_loss_(latent)

        return (
            (rec_loss + self.hparams.lda * reg_loss),
            rec_loss,
            reg_loss,
        )

    def reg_loss_(self, latent):
        latent = torch.reshape(latent, (-1, self.hparams.num_frames - 3, latent.shape[-1]))
        means = torch.mean(latent, dim=(0,1), keepdim=True)
        stds = torch.std(latent, dim=(0,1), keepdim=True)
        scaled_latent = (latent - means) / (stds + 1e-7)
        deriv_1 = scaled_latent[:, 1:] - scaled_latent[:, :-1]
        deriv_2 = deriv_1[:, 1:] - deriv_1[:, :-1]
        deriv_3 = deriv_2[:, 1:] - deriv_2[:, :-1]
        deriv_4 = deriv_3[:, 1:] - deriv_3[:, :-1]
        deriv_1_loss = torch.mean(torch.sum(torch.square(deriv_1), dim=2))
        deriv_2_loss = torch.mean(torch.sum(torch.square(deriv_2), dim=2))
        deriv_3_loss = torch.mean(torch.sum(torch.square(deriv_3), dim=2))
        deriv_4_loss = torch.mean(torch.sum(torch.square(deriv_4), dim=2))
        reg_loss = deriv_1_loss + deriv_2_loss + deriv_3_loss + deriv_4_loss
        return reg_loss

    def train_forward(self, x):
        if self.hparams.model_name == 'encoder-decoder-64':
            output, latent, mu, logvar = self.model(x, None)
            return output, latent, mu, logvar
        elif self.hparams.model_name == 'refine-64':
            output, latent = self.model(x)
            return output, latent
        output = torch.reshape(output, x.size())
        latent = torch.reshape(latent, (*x.size()[:2], -1))
        return output, latent

    def training_step(self, batch, batch_idx):
        data, target, filepath = batch
        if self.hparams.model_name == 'encoder-decoder-64':
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            target = torch.reshape(target, (-1, *target.shape[-3:]))
            filepath = np.transpose(filepath).flatten()
            output, latent, mu, logvar = self.train_forward(data)

            train_loss, train_rec_loss, train_reg_loss, train_kl_loss = self.encoder_decoder_loss(output, target, mu, logvar)
            self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train_rec_loss', train_rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train_reg_loss', train_reg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train_kl_loss', train_kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.global_iter += 1

        elif self.hparams.model_name == 'refine-64':
            mu, logvar = data[:, :, :, 0], data[:, :, :, 1]
            mu = torch.reshape(mu, (-1, mu.shape[-1]))
            logvar = torch.reshape(logvar, (-1, logvar.shape[-1]))
            target = torch.reshape(target, (-1, target.shape[-1]))
            filepath = np.transpose(filepath).flatten()
            output, latent = self.train_forward(mu)
        
            train_loss, train_rec_loss, train_reg_loss = self.refine_loss(output, latent, mu, logvar)
            self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train_rec_loss', train_rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train_reg_loss', train_reg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        data, target, filepath = batch
        if self.hparams.model_name == 'encoder-decoder-64':
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            target = torch.reshape(target, (-1, *target.shape[-3:]))
            filepath = np.transpose(filepath).flatten()
            output, latent, mu, logvar = self.train_forward(data)
        
            val_loss, val_rec_loss, val_reg_loss, val_kl_loss = self.encoder_decoder_loss(output, target, mu, logvar)
            self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.val_batch, sync_dist=True)
            self.log('val_rec_loss', val_rec_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.val_batch, sync_dist=True)
            self.log('val_reg_loss', val_reg_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.val_batch, sync_dist=True)
            self.log('val_kl_loss', val_kl_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
        
        elif self.hparams.model_name == 'refine-64':
            mu, logvar = data[:, :, :, 0], data[:, :, :, 1]
            mu = torch.reshape(mu, (-1, mu.shape[-1]))
            logvar = torch.reshape(logvar, (-1, logvar.shape[-1]))
            target = torch.reshape(target, (-1, target.shape[-1]))
            filepath = np.transpose(filepath).flatten()
            output, latent = self.train_forward(mu)

            val_loss, val_rec_loss, val_reg_loss = self.refine_loss(output, latent, mu, logvar)
            self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.val_batch, sync_dist=True)
            self.log('val_rec_loss', val_rec_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.val_batch, sync_dist=True)
            self.log('val_reg_loss', val_reg_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.val_batch, sync_dist=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        
        if self.hparams.model_name == 'encoder-decoder-64':
            data, target, filepath = batch
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            target = torch.reshape(target, (-1, *target.shape[-3:]))
            filepath = np.transpose(filepath).flatten()
            output, latent, mu, logvar = self.train_forward(data)

            test_loss, test_rec_loss, test_reg_loss, test_kl_loss = self.encoder_decoder_loss(output, target, mu, logvar)
            self.log('test_loss', test_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.test_batch, sync_dist=True)
            self.log('test_rec_loss', test_rec_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.test_batch, sync_dist=True)
            self.log('test_reg_loss', test_reg_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.test_batch, sync_dist=True)
            self.log('test_kl_loss', test_kl_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.test_batch, sync_dist=True)
            self.all_filepaths.extend(filepath)
            for idx in range(data.shape[0]):
                comparison = torch.cat([data[idx,:, :, :128].unsqueeze(0),
                                        data[idx,:, :, 128:].unsqueeze(0),
                                        target[idx, :, :, :128].unsqueeze(0),
                                        target[idx, :, :, 128:].unsqueeze(0),
                                        output[idx, :, :, :128].unsqueeze(0),
                                        output[idx, :, :, 128:].unsqueeze(0)])
                save_image(comparison.cpu(), os.path.join(self.pred_log_dir, filepath[idx]), nrow=1)
                latent_tmp = latent[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                self.all_latents.append(latent_tmp)

                mu_tmp = mu[idx].view(1, -1)[0]
                mu_tmp = mu_tmp.cpu().detach().numpy()
                self.all_mus.append(mu_tmp)

                logvar_tmp = logvar[idx].view(1, -1)[0]
                logvar_tmp = logvar_tmp.cpu().detach().numpy()
                self.all_logvars.append(logvar_tmp)

        elif self.hparams.model_name  == 'refine-64':
            data, target, filepath = batch
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            target = torch.reshape(target, (-1, *target.shape[-3:]))
            filepath = np.transpose(filepath).flatten()

            _, latent, mu, logvar = self.high_dim_model(data, None)
            latent_reconstructed, latent_latent = self.model(mu.squeeze())
            output, _, _, _ = self.high_dim_model(data, latent_reconstructed)
            output = torch.reshape(output, target.size())
            mse_loss = torch.nn.MSELoss(reduction='sum')
            # latent_reconstruction_loss = mse_loss(latent_reconstructed, mu) / target.size()[0]
            _, latent_reconstruction_loss, _ = self.refine_loss(latent_reconstructed, mu, mu, logvar)
            pixel_reconstruction_loss = mse_loss(output, target) / target.size()[0]
            self.log('latent_reconstruction_loss', latent_reconstruction_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.test_batch, sync_dist=True)
            self.log('pixel_reconstruction_loss', pixel_reconstruction_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.test_batch, sync_dist=True)
            self.all_filepaths.extend(filepath)
            for idx in range(data.shape[0]):
                comparison = torch.cat([data[idx,:, :, :128].unsqueeze(0),
                                        data[idx,:, :, 128:].unsqueeze(0),
                                        target[idx, :, :, :128].unsqueeze(0),
                                        target[idx, :, :, 128:].unsqueeze(0),
                                        output[idx, :, :, :128].unsqueeze(0),
                                        output[idx, :, :, 128:].unsqueeze(0)])
                save_image(comparison.cpu(), os.path.join(self.pred_log_dir, filepath[idx]), nrow=1)
                # save latent: the latent vector in the encoder-decoder-64 network
                latent_tmp = latent[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                self.all_latents.append(latent_tmp)
                # save latent_latent: the latent vector in the refine network
                latent_latent_tmp = latent_latent[idx].view(1, -1)[0]
                latent_latent_tmp = latent_latent_tmp.cpu().detach().numpy()
                self.all_refine_latents.append(latent_latent_tmp)
                # save latent_reconstructed: the latent vector reconstructed by the entire refine network
                latent_reconstructed_tmp = latent_reconstructed[idx].view(1, -1)[0]
                latent_reconstructed_tmp = latent_reconstructed_tmp.cpu().detach().numpy()
                self.all_reconstructed_latents.append(latent_reconstructed_tmp)

    def test_save(self):
        if self.hparams.model_name == 'encoder-decoder-64':
            np.save(os.path.join(self.var_log_dir, 'ids.npy'), self.all_filepaths)
            np.save(os.path.join(self.var_log_dir, 'latent.npy'), self.all_latents)
            np.save(os.path.join(self.var_log_dir, 'mu.npy'), self.all_mus)
            np.save(os.path.join(self.var_log_dir, 'logvar.npy'), self.all_logvars)
        elif self.hparams.model_name in 'refine-64':
            np.save(os.path.join(self.var_log_dir, 'ids.npy'), self.all_filepaths)
            np.save(os.path.join(self.var_log_dir, 'latent.npy'), self.all_latents)
            np.save(os.path.join(self.var_log_dir, 'refine_latent.npy'), self.all_refine_latents)
            np.save(os.path.join(self.var_log_dir, 'reconstructed_latent.npy'), self.all_reconstructed_latents)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
    
    def paths_to_tuple(self, paths):
        new_paths = []
        for i in range(len(paths)):
            tmp = paths[i].split('.')[0].split('_')
            new_paths.append((int(tmp[0]), int(tmp[1])))
        return new_paths

    def setup(self, stage=None):

        if stage == 'fit':
            if self.hparams.model_name == 'encoder-decoder-64':
                self.train_dataset = NeuralPhysDataset(data_filepath=self.hparams.data_filepath,
                                                       num_frames=self.hparams.num_frames,
                                                       flag='train',
                                                       seed=self.hparams.seed,
                                                       object_name=self.hparams.dataset)
                self.val_dataset =   NeuralPhysDataset(data_filepath=self.hparams.data_filepath,
                                                       num_frames=self.hparams.num_frames,
                                                       flag='val',
                                                       seed=self.hparams.seed,
                                                       object_name=self.hparams.dataset)
            if self.hparams.model_name == 'refine-64':
                high_dim_var_log_dir = self.var_log_dir.replace('refine', 'encoder-decoder')
                train_mu = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_train', 'mu.npy')))
                train_logvar = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_train', 'logvar.npy')))
                train_mu = torch.unsqueeze(train_mu, dim=2)
                train_logvar = torch.unsqueeze(train_logvar, dim=2)
                train_data = torch.cat([train_mu, train_logvar], axis=2)
                train_target = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_train', 'mu.npy')))

                val_mu = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_val', 'mu.npy')))
                val_logvar = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_val', 'logvar.npy')))
                val_target = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_val', 'mu.npy')))
                val_mu = torch.unsqueeze(val_mu, dim=2)
                val_logvar = torch.unsqueeze(val_logvar, dim=2)
                val_data = torch.cat([val_mu, val_logvar], axis=2)
                val_target = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_val', 'mu.npy')))

                train_filepaths = list(np.load(os.path.join(high_dim_var_log_dir+'_train', 'ids.npy')))
                val_filepaths = list(np.load(os.path.join(high_dim_var_log_dir+'_val', 'ids.npy')))
                # convert the file strings into tuple so that we can use TensorDataset to load everything together
                train_filepaths = torch.Tensor(self.paths_to_tuple(train_filepaths))
                val_filepaths = torch.Tensor(self.paths_to_tuple(val_filepaths))
                self.train_dataset = NeuralPhysRefineDataset(data=train_data,
                                                             target=train_target,
                                                             filepaths=train_filepaths,
                                                             num_frames=self.hparams.num_frames)
                self.val_dataset =   NeuralPhysRefineDataset(data=val_data,
                                                             target=val_target,
                                                             filepaths=val_filepaths,
                                                             num_frames=self.hparams.num_frames)

        if stage == 'test':
            self.test_dataset = NeuralPhysDataset(data_filepath=self.hparams.data_filepath,
                                                  num_frames=self.hparams.num_frames,
                                                  flag='test',
                                                  seed=self.hparams.seed,
                                                  object_name=self.hparams.dataset)
            # initialize lists for saving variables and latents during testing
            self.all_filepaths = []
            self.all_latents = []
            self.all_mus = []
            self.all_logvars = []
            self.all_refine_latents = []
            self.all_reconstructed_latents = []

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.hparams.train_batch,
                                                   shuffle=True,
                                                   **self.kwargs)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                 batch_size=self.hparams.val_batch,
                                                 shuffle=False,
                                                 **self.kwargs)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.hparams.test_batch,
                                                  shuffle=False,
                                                  **self.kwargs)
        return test_loader