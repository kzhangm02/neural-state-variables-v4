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
                         EncoderDecoderDynamicsNetwork,
                         RefineCircularMotionModel,
                         CircularMotionDynamicsModel,
                         RefineSinglePendulumModel,
                         SinglePendulumDynamicsModel,
                         RefineDoublePendulumModel,
                         DoublePendulumDynamicsModel,
                         RefineElasticPendulumModel)

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def rename_ckpt_for_multi_models(ckpt):
    renamed_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if 'dynamics' in k:
            continue
        if 'high_dim_model' in k:
            continue
        else:
            name = k.replace('model.', '')
        renamed_state_dict[name] = v
    return renamed_state_dict

def load_checkpoint(filepath, model):
    checkpoint_filepath = glob.glob(os.path.join(filepath, '*.ckpt'))[0]
    print('Loading', checkpoint_filepath)
    ckpt = torch.load(checkpoint_filepath)
    ckpt = rename_ckpt_for_multi_models(ckpt)
    model.load_state_dict(ckpt)

class VisDynamicsModel(pl.LightningModule):

    def __init__(
        self,
        lda1: float=1.0,
        lda2: float=1.0,
        lda3: float=1.0,
        beta: float=1.0,
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
        lr_schedule: list=[],
        high_dim_ckpt_dir: str='',
    ) -> None:
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
            self.dynamics_model = EncoderDecoderDynamicsNetwork(in_channels=64)
        if self.hparams.model_name == 'refine-64':
            self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)
            load_checkpoint(self.hparams.high_dim_ckpt_dir, self.high_dim_model)
            for name, param in self.high_dim_model.named_parameters():
                param.requires_grad = False
            if self.hparams.dataset == 'circular_motion':
                self.model = RefineCircularMotionModel(in_channels=64)
                self.dynamics_model = CircularMotionDynamicsModel(in_channels=2)
            if self.hparams.dataset == 'single_pendulum':
                self.model = RefineSinglePendulumModel(in_channels=64)
                self.dynamics_model = CircularMotionDynamicsModel(in_channels=2)
            if self.hparams.dataset == 'double_pendulum':
                self.model = RefineDoublePendulumModel(in_channels=64)
                self.dynamics_model = CircularMotionDynamicsModel(in_channels=4)
            if self.hparams.dataset == 'elastic_pendulum':
                self.model = RefineElasticPendulumModel(in_channels=64)
            # if self.hparams.if_test:
            #     self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)

    def encoder_decoder64_loss(
        self, 
        target, 
        output, 
        next_output, 
        latent, 
        next_latent, 
        mu, 
        logvar
    ):
        B, F = target.shape[0], target.shape[1]
        output = torch.reshape(output, target.shape)
        next_output = torch.reshape(next_output, target.shape)
        latent = torch.reshape(latent, (B, F, 64))
        next_latent = torch.reshape(next_latent, (B, F, 64))
        mu = torch.reshape(mu, (B, F, 64))
        logvar = torch.reshape(logvar, (B, F, 64))

        mse_loss = torch.nn.MSELoss(reduction='sum')
        rec_loss = mse_loss(output, target) / (B*F) + mse_loss(next_output[:, :-1], target[:, 1:]) / (B*(F-1))
        nll_loss = logvar[:, :-1] + (next_latent[:, 1:] - mu[:, :-1]) ** 2 / logvar[:, :-1].exp()
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (B*F)
        latent_rec_loss = torch.sum(nll_loss) / (2*B*(F-1))
        reg_loss = self.smoothness_loss(mu)

        return (
            (
                rec_loss + \
                self.hparams.lda1 * reg_loss + \
                self.hparams.lda2 * latent_rec_loss + \
                self.hparams.beta * kl_loss
            ),
            rec_loss, 
            reg_loss,
            latent_rec_loss,
            kl_loss, 
        )
    
    def refine_loss(
        self, 
        target, 
        output, 
        next_output, 
        mu, 
        logvar, 
        reconstructed_latent, 
        next_reconstructed_latent, 
        latent_latent, 
        next_latent_latent
    ):
        B, F = target.shape[0], target.shape[1]
        output = torch.reshape(output, target.shape)
        next_output = torch.reshape(next_output, target.shape)
        mu = torch.reshape(mu, (B, F, 64))
        logvar = torch.reshape(logvar, (B, F, 64))
        reconstructed_latent = torch.reshape(reconstructed_latent, mu.shape)
        next_reconstructed_latent = torch.reshape(next_reconstructed_latent, mu.shape)
        latent_latent = torch.reshape(latent_latent, (B, F, latent_latent.shape[-1]))
        next_latent_latent = torch.reshape(next_latent_latent, (B, F, next_latent_latent.shape[-1]))

        mse_loss = torch.nn.MSELoss(reduction='sum')
        rec_loss = mse_loss(output, target) / (B*F) + mse_loss(next_output[:, :-1], target[:, 1:]) / (B*(F-1))
        latent_rec_loss = mse_loss(latent_latent[1:], next_latent_latent[:-1]) / (B*(F-1))
        nll_loss = logvar + (reconstructed_latent - mu) ** 2 / logvar.exp()
        latent64_rec_loss = torch.sum(nll_loss) / (2*B*F)
        next_nll_loss = logvar[:, :-1] + (reconstructed_latent[:, 1:] - mu[:, :-1]) ** 2 / logvar[:, :-1].exp()
        latent64_rec_loss += torch.sum(next_nll_loss) / (2*B*(F-1))
        reg_loss = self.smoothness_loss(latent_latent)

        return (
            (
                rec_loss + \
                self.hparams.lda1 * reg_loss + \
                self.hparams.lda2 * latent64_rec_loss + \
                self.hparams.lda3 * latent_rec_loss
            ),
            rec_loss, 
            reg_loss,
            latent64_rec_loss,
            latent_rec_loss, 
        )

    def smoothness_loss(self, latent):
        loss = 0
        derivs = 4
        maxs = torch.amax(latent, dim=1, keepdim=True)
        mins = torch.amin(latent, dim=1, keepdim=True)
        scaled_latent = latent / (maxs - mins + 1e-12)
        last_deriv = scaled_latent
        for k in range(derivs):
            deriv = last_deriv[:, 1:] - last_deriv[:, :-1]
            deriv_loss = torch.mean(torch.sum(torch.abs(deriv), dim=1))   # sum along time dimension
            loss += (5.0**(k+1)) * deriv_loss
            last_deriv = deriv
        return loss

    def training_step(self, batch, batch_idx):
        data, target, filepath = batch
        if self.hparams.model_name == 'encoder-decoder-64':
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            output, latent, mu, logvar = self.model(data, None)
            next_latent = self.dynamics_model(mu)
            next_output = self.model.decoder(next_latent)

            losses = self.encoder_decoder64_loss(target, output, next_output, latent, next_latent, mu, logvar)
            train_loss, train_rec_loss, train_reg_loss, train_latent_rec_loss, train_kl_loss = losses
            self.log('train/loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/rec_loss', train_rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/reg_loss', train_reg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/latent_rec_loss', train_latent_rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/kl_loss', train_kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.global_iter += 1

        elif self.hparams.model_name == 'refine-64':
            self.high_dim_model.eval()
            mu, logvar = data[:, :, :, 0], data[:, :, :, 1] 
            mu = torch.reshape(mu, (-1, mu.shape[-1]))
            reconstructed_latent, latent_latent = self.model(mu)
            next_latent_latent = self.dynamics_model(latent_latent)
            next_reconstructed_latent = self.model.decoder(next_latent_latent)
            next_output = self.high_dim_model.decoder(next_reconstructed_latent)
            output = self.high_dim_model.decoder(reconstructed_latent)

            losses = self.refine_loss(target, output, next_output, mu, logvar, reconstructed_latent, next_reconstructed_latent, latent_latent, next_latent_latent)
            train_loss, train_rec_loss, train_reg_loss, train_latent64_rec_loss, train_latent_rec_loss = losses 
            self.log('train/loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/rec_loss', train_rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/reg_loss', train_reg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/latent64_rec_loss', train_latent64_rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/latent_rec_loss', train_latent_rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        data, target, filepath = batch
        if self.hparams.model_name == 'encoder-decoder-64':
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            output, latent, mu, logvar = self.model(data, None)
            next_latent = self.dynamics_model(mu)
            next_output = self.model.decoder(next_latent)

            losses = self.encoder_decoder64_loss(target, output, next_output, latent, next_latent, mu, logvar)
            val_loss, val_rec_loss, val_reg_loss, val_latent_rec_loss, val_kl_loss = losses
            self.log('val/loss', val_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/rec_loss', val_rec_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/reg_loss', val_reg_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/latent_rec_loss', val_latent_rec_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/kl_loss', val_kl_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.global_iter += 1
        
        elif self.hparams.model_name == 'refine-64':
            self.high_dim_model.eval()
            mu, logvar = data[:, :, :, 0], data[:, :, :, 1] 
            mu = torch.reshape(mu, (-1, mu.shape[-1]))
            reconstructed_latent, latent_latent = self.model(mu)
            next_latent_latent = self.dynamics_model(latent_latent)
            next_reconstructed_latent = self.model.decoder(next_latent_latent)
            next_output = self.high_dim_model.decoder(next_reconstructed_latent)
            output = self.high_dim_model.decoder(reconstructed_latent)

            losses = self.refine_loss(target, output, next_output, mu, logvar, reconstructed_latent, next_reconstructed_latent, latent_latent, next_latent_latent)
            val_loss, val_rec_loss, val_reg_loss, val_latent64_rec_loss, val_latent_rec_loss = losses 
            self.log('val/loss', val_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/rec_loss', val_rec_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/reg_loss', val_reg_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/latent64_rec_loss', val_latent64_rec_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/latent_rec_loss', val_latent_rec_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
        
        return val_loss

    def test_step(self, batch, batch_idx):
        
        if self.hparams.model_name == 'encoder-decoder-64':
            data, target, filepath = batch
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            output, latent, mu, logvar = self.model(data, None)
            next_latent = self.dynamics_model(mu)
            next_output = self.model.decoder(next_latent)

            losses = self.encoder_decoder64_loss(target, output, next_output, latent, next_latent, mu, logvar)
            test_loss, test_rec_loss, test_reg_loss, test_latent_rec_loss, test_kl_loss = losses
            self.log('test/loss', test_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/rec_loss', test_rec_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/reg_loss', test_reg_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/latent_rec_loss', test_latent_rec_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/kl_loss', test_kl_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            
            target = torch.reshape(target, (-1, *target.shape[-3:]))
            filepath = np.transpose(filepath).flatten()
            self.all_filepaths.extend(filepath)
            for idx in range(data.shape[0]):
                comparison = torch.cat([
                    target[idx, :, :, :128].unsqueeze(0),
                    target[idx, :, :, 128:].unsqueeze(0),
                    output[idx, :, :, :128].unsqueeze(0),
                    output[idx, :, :, 128:].unsqueeze(0),
                    next_output[idx, :, :, :128].unsqueeze(0),
                    next_output[idx, :, :, 128:].unsqueeze(0)
                ])
                save_image(comparison.cpu(), os.path.join(self.pred_log_dir, filepath[idx]), nrow=1)
                # save latent: the latent vector in the encoder-decoder-64 network
                latent_tmp = latent[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                self.all_latents.append(latent_tmp)
                # save mu: the mu vector in the encoder-decoder-64 network
                mu_tmp = mu[idx].view(1, -1)[0]
                mu_tmp = mu_tmp.cpu().detach().numpy()
                self.all_mus.append(mu_tmp)
                # save logvar: the log of variance vector in the encoder-decoder-64 network
                logvar_tmp = logvar[idx].view(1, -1)[0]
                logvar_tmp = logvar_tmp.cpu().detach().numpy()
                self.all_logvars.append(logvar_tmp)
                # save next_latent: the next latent vector predicted by the dynamics network
                next_latent_tmp = next_latent[idx].view(1, -1)[0]
                next_latent_tmp = next_latent_tmp.cpu().detach().numpy()
                self.all_next_latents.append(next_latent_tmp)

        elif self.hparams.model_name  == 'refine-64':
            self.high_dim_model.eval()
            data, target, filepath = batch
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            _, _, mu, logvar = self.high_dim_model(data, None)
            reconstructed_latent, latent_latent = self.model(mu)
            next_latent_latent = self.dynamics_model(latent_latent)
            next_reconstructed_latent = self.model.decoder(next_latent_latent)
            next_output = self.high_dim_model.decoder(next_reconstructed_latent)
            output = self.high_dim_model.decoder(reconstructed_latent)

            losses = self.refine_loss(target, output, next_output, mu, logvar, reconstructed_latent, next_reconstructed_latent, latent_latent, next_latent_latent)
            test_loss, test_rec_loss, test_reg_loss, test_latent64_rec_loss, test_latent_rec_loss = losses 
            self.log('test/loss', test_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/rec_loss', test_rec_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/reg_loss', test_reg_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/latent64_rec_loss', test_latent64_rec_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/latent_rec_loss', test_latent_rec_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            
            target = torch.reshape(target, (-1, *target.shape[-3:]))
            filepath = np.transpose(filepath).flatten()
            self.all_filepaths.extend(filepath)
            for idx in range(data.shape[0]):
                comparison = torch.cat([
                    target[idx, :, :, :128].unsqueeze(0),
                    target[idx, :, :, 128:].unsqueeze(0),
                    output[idx, :, :, :128].unsqueeze(0),
                    output[idx, :, :, 128:].unsqueeze(0),
                    next_output[idx, :, :, :128].unsqueeze(0),
                    next_output[idx, :, :, 128:].unsqueeze(0)
                ])
                save_image(comparison.cpu(), os.path.join(self.pred_log_dir, filepath[idx]), nrow=1)
                # save latent: the latent vector in the encoder-decoder-64 network
                latent_tmp = mu[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                self.all_latents.append(latent_tmp)
                # save latent_latent: the latent vector in the refine network
                latent_latent_tmp = latent_latent[idx].view(1, -1)[0]
                latent_latent_tmp = latent_latent_tmp.cpu().detach().numpy()
                self.all_refine_latents.append(latent_latent_tmp)
                # save reconstructed_latent: the latent vector reconstructed by the refine network
                reconstructed_latent_tmp = reconstructed_latent[idx].view(1, -1)[0]
                reconstructed_latent_tmp = reconstructed_latent_tmp.cpu().detach().numpy()
                self.all_reconstructed_latents.append(reconstructed_latent_tmp)
                # save next_latent_latent: the next latent vector predicted by the refine dynamics network
                next_latent_latent_tmp = next_latent_latent[idx].view(1, -1)[0]
                next_latent_latent_tmp = next_latent_latent_tmp.cpu().detach().numpy()
                self.all_next_refine_latents.append(next_latent_latent_tmp)

    def test_save(self):
        if self.hparams.model_name == 'encoder-decoder-64':
            np.save(os.path.join(self.var_log_dir, 'ids.npy'), self.all_filepaths)
            np.save(os.path.join(self.var_log_dir, 'latent.npy'), self.all_latents)
            np.save(os.path.join(self.var_log_dir, 'mu.npy'), self.all_mus)
            np.save(os.path.join(self.var_log_dir, 'logvar.npy'), self.all_logvars)
            np.save(os.path.join(self.var_log_dir, 'next_latent.npy'), self.all_next_latents)
        elif self.hparams.model_name in 'refine-64':
            np.save(os.path.join(self.var_log_dir, 'ids.npy'), self.all_filepaths)
            np.save(os.path.join(self.var_log_dir, 'latent.npy'), self.all_latents)
            np.save(os.path.join(self.var_log_dir, 'refine_latent.npy'), self.all_refine_latents)
            np.save(os.path.join(self.var_log_dir, 'next_refine_latent.npy'), self.all_next_refine_latents)
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
                self.train_dataset = NeuralPhysDataset(
                    data_filepath=self.hparams.data_filepath,
                    num_frames=self.hparams.num_frames,
                    flag='train',
                    seed=self.hparams.seed,
                    object_name=self.hparams.dataset,
                )
                self.val_dataset = NeuralPhysDataset(
                    data_filepath=self.hparams.data_filepath,
                    num_frames=self.hparams.num_frames,
                    flag='val',
                    seed=self.hparams.seed,
                    object_name=self.hparams.dataset,
                )
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
                self.train_dataset = NeuralPhysRefineDataset(
                    data_filepath=self.hparams.data_filepath,
                    object_name=self.hparams.dataset,
                    data=train_data,
                    target=train_target,
                    filepaths=train_filepaths,
                    num_frames=self.hparams.num_frames
                )
                self.val_dataset = NeuralPhysRefineDataset(
                    data_filepath=self.hparams.data_filepath,
                    object_name=self.hparams.dataset,
                    data=val_data,
                    target=val_target,
                    filepaths=val_filepaths,
                    num_frames=self.hparams.num_frames
                )

        if stage == 'test':
            self.test_dataset = NeuralPhysDataset(
                data_filepath=self.hparams.data_filepath,
                num_frames=self.hparams.num_frames,
                flag='test',
                seed=self.hparams.seed,
                object_name=self.hparams.dataset
            )
            # initialize lists for saving variables and latents during testing
            self.all_filepaths = []
            self.all_latents = []
            self.all_mus = []
            self.all_logvars = []
            self.all_next_latents = []
            self.all_refine_latents = []
            self.all_next_refine_latents = []
            self.all_reconstructed_latents = []

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.train_batch,
            shuffle=True,
            **self.kwargs
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.val_batch,
            shuffle=False,
            **self.kwargs
        )
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.test_batch,
            shuffle=False,
            **self.kwargs
        )
        return test_loader