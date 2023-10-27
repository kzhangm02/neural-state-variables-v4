import os
import sys
import glob
import yaml
import torch
import pprint
import shutil
import numpy as np
from tqdm import tqdm
from munch import munchify
from pysr import PySRRegressor
from collections import OrderedDict
from models import VisDynamicsModel
from dataset import NeuralPhysDataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)

def main():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)
    log_dir_name = '_'.join([cfg.dataset, cfg.model_name, str(cfg.seed)])
    log_dir = os.path.join(cfg.log_dir, log_dir_name)
    high_dim_checkpoint_filepath = str(sys.argv[2])
    refine_checkpoint_filepath = str(sys.argv[3])

    model = VisDynamicsModel(
        beta=cfg.beta,
        lda1=cfg.lda1,
        lda2=cfg.lda2,
        lda3=cfg.lda3,
        lr=cfg.lr,
        seed=cfg.seed,
        if_cuda=cfg.if_cuda,
        if_test=True,
        gamma=cfg.gamma,
        log_dir=log_dir,
        train_batch=cfg.train_batch,
        val_batch=cfg.val_batch,
        test_batch=cfg.test_batch,
        num_workers=cfg.num_workers,
        model_name=cfg.model_name,
        data_filepath=cfg.data_filepath,
        dataset=cfg.dataset,
        num_frames=cfg.num_frames,
        lr_schedule=cfg.lr_schedule,
        high_dim_ckpt_dir=high_dim_checkpoint_filepath
    )
    
    def load_checkpoint(filepath, model):
        checkpoint_filepath = glob.glob(os.path.join(filepath, '*.ckpt'))[0]
        ckpt = torch.load(checkpoint_filepath)
        ckpt = rename_ckpt_for_multi_models(ckpt)
        model.load_state_dict(ckpt)
    
    def load_dynamics_checkpoint(filepath, model):
        checkpoint_filepath = glob.glob(os.path.join(filepath, '*.ckpt'))[0]
        ckpt = torch.load(checkpoint_filepath)
        ckpt = rename_ckpt_for_dynamics_models(ckpt)
        model.load_state_dict(ckpt)

    if cfg.model_name == 'encoder-decoder-64':
        load_checkpoint(high_dim_checkpoint_filepath, model.model)
        load_dynamics_checkpoint(high_dim_checkpoint_filepath, model.dynamics_model)
    elif cfg.model_name == 'refine-64':
        load_checkpoint(refine_checkpoint_filepath, model.model)
        load_dynamics_checkpoint(refine_checkpoint_filepath, model.dynamics_model)

    model.eval()
    model.freeze()
    trainer = Trainer(
        accelerator='gpu',
        num_nodes=cfg.num_gpus,
        deterministic=True,
        strategy='ddp',
        default_root_dir=log_dir,
        val_check_interval=1.0,
    )

    trainer.test(model)
    model.test_save()

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

def rename_ckpt_for_dynamics_models(ckpt):
    renamed_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if 'dynamics' in k:
            name = k.replace('dynamics_model.', '')
        else:
            continue
        renamed_state_dict[name] = v
    return renamed_state_dict

# gather latent variables by running training data on the trained high-dim models
def gather_latent_from_trained_high_dim_model():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)
    log_dir_name = '_'.join([cfg.dataset, cfg.model_name, str(cfg.seed)])
    log_dir = os.path.join(cfg.log_dir, log_dir_name)

    model = VisDynamicsModel(
        beta=cfg.beta,
        lda1=cfg.lda1,
        lda2=cfg.lda2,
        lda3=cfg.lda3,
        lr=cfg.lr,
        seed=cfg.seed,
        if_cuda=cfg.if_cuda,
        if_test=True,
        gamma=cfg.gamma,
        log_dir=log_dir,
        train_batch=cfg.train_batch,
        val_batch=cfg.val_batch,
        test_batch=cfg.test_batch,
        num_workers=cfg.num_workers,
        model_name=cfg.model_name,
        data_filepath=cfg.data_filepath,
        dataset=cfg.dataset,
        num_frames=cfg.num_frames,
        lr_schedule=cfg.lr_schedule
    )

    high_dim_checkpoint_filepath = str(sys.argv[2])
    high_dim_checkpoint_filepath = glob.glob(os.path.join(high_dim_checkpoint_filepath, '*.ckpt'))[0]
    high_dim_ckpt = torch.load(high_dim_checkpoint_filepath)
    model.load_state_dict(high_dim_ckpt['state_dict'])

    model = model.to('cuda')
    model.eval()
    model.freeze()

    # prepare train and val dataset
    kwargs = {'num_workers': cfg.num_workers, 'pin_memory': True} if cfg.if_cuda else {}
    train_dataset = NeuralPhysDataset(
        data_filepath=cfg.data_filepath,
        num_frames=cfg.num_frames,
        flag='train',
        seed=cfg.seed,
        object_name=cfg.dataset
    )
    val_dataset = NeuralPhysDataset(
        data_filepath=cfg.data_filepath,
        num_frames=cfg.num_frames,
        flag='val',
        seed=cfg.seed,
        object_name=cfg.dataset
    )
    # prepare train and val loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train_batch,
        shuffle=False,
        **kwargs
    )
    val_loader =   torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.val_batch,
        shuffle=False,
        **kwargs
    )
    
    def gather_latent(flag):
        if flag == 'train':
            loader = train_loader
        elif flag == 'val':
            loader = val_loader
        all_filepaths = []
        all_latents, all_mus, all_logvars, all_next_latent = [], [], [], []
        var_log_dir = os.path.join(log_dir, 'variables')
        for batch_idx, (data, target, filepath) in enumerate(tqdm(loader)):
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            target = torch.reshape(target, (-1, *target.shape[-3:]))
            filepath = np.transpose(filepath).flatten()
            _, latent, mu, logvar = model.model(data.cuda(), None)
            next_latent = model.dynamics_model(latent)

            latent = torch.reshape(latent, (data.size()[0], -1))
            mu = torch.reshape(mu, (data.size()[0], -1))
            logvar = torch.reshape(logvar, (data.size()[0], -1))
            next_latent = torch.reshape(next_latent, (data.size()[0], -1))

            # save the latent vectors
            all_filepaths.extend(filepath)
            for idx in range(data.shape[0]):
                latent_tmp = latent[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                all_latents.append(latent_tmp)

                mu_tmp = mu[idx].view(1, -1)[0]
                mu_tmp = mu_tmp.cpu().detach().numpy()
                all_mus.append(mu_tmp)

                logvar_tmp = logvar[idx].view(1, -1)[0]
                logvar_tmp = logvar_tmp.cpu().detach().numpy()
                all_logvars.append(logvar_tmp)

                next_latent_tmp = next_latent[idx].view(1, -1)[0]
                next_latent_tmp = next_latent_tmp.cpu().detach().numpy()
                all_next_latent.append(next_latent_tmp)

        mkdir(var_log_dir+'_'+flag)
        np.save(os.path.join(var_log_dir+'_'+flag, 'ids.npy'), all_filepaths)
        np.save(os.path.join(var_log_dir+'_'+flag, 'latent.npy'), all_latents)
        np.save(os.path.join(var_log_dir+'_'+flag, 'mu.npy'), all_mus)
        np.save(os.path.join(var_log_dir+'_'+flag, 'logvar.npy'), all_logvars)
        np.save(os.path.join(var_log_dir+'_'+flag, 'next_latent.npy'), all_next_latent)

    # run train forward pass to save the latent vector for training the refine network later
    gather_latent(flag='train')
                        
    # run val forward pass to save the latent vector for validating the refine network later
    gather_latent(flag='val')

# gather latent variables by running training data on the trained refine models
def gather_latent_from_trained_refine_model():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)
    log_dir_name = '_'.join([cfg.dataset, cfg.model_name, str(cfg.seed)])
    log_dir = os.path.join(cfg.log_dir, log_dir_name)
    high_dim_checkpoint_filepath = str(sys.argv[2])
    refine_checkpoint_filepath = str(sys.argv[3])

    model = VisDynamicsModel(
        beta=cfg.beta,
        lda1=cfg.lda1,
        lda2=cfg.lda2,
        lda3=cfg.lda3,
        lr=cfg.lr,
        seed=cfg.seed,
        if_cuda=cfg.if_cuda,
        if_test=True,
        gamma=cfg.gamma,
        log_dir=log_dir,
        train_batch=cfg.train_batch,
        val_batch=cfg.val_batch,
        test_batch=cfg.test_batch,
        num_workers=cfg.num_workers,
        model_name=cfg.model_name,
        data_filepath=cfg.data_filepath,
        dataset=cfg.dataset,
        lr_schedule=cfg.lr_schedule,
        high_dim_ckpt_dir=high_dim_checkpoint_filepath
    )

    refine_checkpoint_filepath = glob.glob(os.path.join(refine_checkpoint_filepath, '*.ckpt'))[0]
    print('Loading', refine_checkpoint_filepath)
    refine_ckpt = torch.load(refine_checkpoint_filepath)
    ckpt = rename_ckpt_for_multi_models(refine_ckpt)
    model.model.load_state_dict(ckpt)
    ckpt = rename_ckpt_for_dynamics_models(refine_ckpt)
    model.dynamics_model.load_state_dict(ckpt)

    model = model.to('cuda')
    model.eval()
    model.freeze()

    # prepare train and val dataset
    kwargs = {'num_workers': cfg.num_workers, 'pin_memory': True} if cfg.if_cuda else {}
    train_dataset = NeuralPhysDataset(
        data_filepath=cfg.data_filepath,
        num_frames=cfg.num_frames,
        flag='train',
        seed=cfg.seed,
        object_name=cfg.dataset
    )
    val_dataset =   NeuralPhysDataset(
        data_filepath=cfg.data_filepath,
        num_frames=cfg.num_frames,
        flag='val',
        seed=cfg.seed,
        object_name=cfg.dataset
    )
    # prepare train and val loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.gather_train_batch,
        shuffle=False,
        **kwargs
    )
    val_loader =   torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.gather_val_batch,
        shuffle=False,
        **kwargs
    )
    
    def gather_latent(flag):
        if flag == 'train':
            loader = train_loader
        elif flag == 'val':
            loader = val_loader
        all_filepaths = []
        all_latents = []
        all_mus = []
        all_logvars = []
        all_refine_latents = []
        all_next_refine_latents = []
        all_reconstructed_latents = []
        var_log_dir = os.path.join(log_dir, 'variables')
        for batch_idx, (data, target, filepath) in enumerate(tqdm(loader)):
            data = data.cuda()
            target = target.cuda()
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            target = torch.reshape(target, (-1, *target.shape[-3:]))
            filepath = np.transpose(filepath).flatten()
            _, latent64, mu64, logvar64 = model.high_dim_model(data, None)

            reconstructed_latent64, latent = model.train_forward(mu64)
            # output = model.high_dim_model.decoder(reconstructed_latent64)
            next_latent = model.dynamics_model(latent)
            # next_reconstructed_latent64 = model.model.decoder(next_latent)
            # next_output = model.high_dim_model.decoder(next_reconstructed_latent64)
            
            # save the latent vectors
            all_filepaths.extend(filepath)
            for idx in range(data.shape[0]):
                # save latent: the latent vector in the encoder-decoder-64 network
                latent_tmp = latent64[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                all_latents.append(latent_tmp)
                # save mu: the mu vector in the encoder-decoder-64 network
                mu_tmp = mu64[idx].view(1, -1)[0]
                mu_tmp = mu_tmp.cpu().detach().numpy()
                all_mus.append(mu_tmp)
                # save logvar: the logvar vector in the encoder-decoder-64 network
                logvar_tmp = logvar64[idx].view(1, -1)[0]
                logvar_tmp = logvar_tmp.cpu().detach().numpy()
                all_logvars.append(logvar_tmp)
                # save refine_latent: the latent vector in the refine network
                refine_latent_tmp = latent[idx].view(1, -1)[0]
                refine_latent_tmp = refine_latent_tmp.cpu().detach().numpy()
                all_refine_latents.append(refine_latent_tmp)
                # save next_refine_latent: the latent vector in the refine network at time t+1
                next_refine_latent_tmp = next_latent[idx].view(1, -1)[0]
                next_refine_latent_tmp = next_refine_latent_tmp.cpu().detach().numpy()
                all_next_refine_latents.append(next_refine_latent_tmp)
                # save latent_reconstructed: the latent vector reconstructed by the entire refine network
                latent_reconstructed_tmp = reconstructed_latent64[idx].view(1, -1)[0]
                latent_reconstructed_tmp = latent_reconstructed_tmp.cpu().detach().numpy()
                all_reconstructed_latents.append(latent_reconstructed_tmp)

        mkdir(var_log_dir+'_'+flag)
        np.save(os.path.join(var_log_dir+'_'+flag, 'ids.npy'), all_filepaths)
        np.save(os.path.join(var_log_dir+'_'+flag, 'latent.npy'), all_latents)
        np.save(os.path.join(var_log_dir+'_'+flag, 'mu.npy'), all_mus)
        np.save(os.path.join(var_log_dir+'_'+flag, 'logvar.npy'), all_logvars)
        np.save(os.path.join(var_log_dir+'_'+flag, 'refine_latent.npy'), all_refine_latents)
        np.save(os.path.join(var_log_dir+'_'+flag, 'next_refine_latent.npy'), all_next_refine_latents)
        np.save(os.path.join(var_log_dir+'_'+flag, 'reconstructed_latent.npy'), all_reconstructed_latents)

    # run train forward pass to save the latent vector for training data
    gather_latent(flag='train')

    # run val forward pass to save the latent vector for validation data
    gather_latent(flag='val')

def eval_sr_model():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    log_dir_name = '_'.join([cfg.dataset, cfg.model_name, str(cfg.seed)])
    log_dir = os.path.join(cfg.log_dir, log_dir_name)

    model_filepath = str(sys.argv[2])
    model = PySRRegressor.from_file(model_filepath)
    all_loss = dict()

    def eval_fit(flag):
        if flag == 'train':
            latent_dir = os.path.join(log_dir, 'variables_train')
        elif flag == 'val':
            latent_dir = os.path.join(log_dir, 'variables_val')
        elif flag == 'test':
            latent_dir = os.path.join(log_dir, 'variables')
        states, latent = load_states_latent(latent_dir)
        latent_fit = model.predict(states)
        loss = np.square(latent - latent_fit).mean()
        all_loss[f'{flag} analytical fit loss'] = loss
        np.save(os.path.join(latent_dir, 'refine_latent_fit.npy'), latent_fit)

    def load_states_latent(latent_dir):
        ids = np.load(os.path.join(latent_dir, 'ids.npy'))
        ids = [int(id[:id.index('_')]) for id in ids][::cfg.num_frames-1]
        states_dir = os.path.join(cfg.data_filepath, 'collect', cfg.dataset, 'states.npy')
        # states = np.load(states_dir)
        # states = states[ids, :cfg.num_frames-1]
        # states = np.reshape(states, (-1, cfg.intrinsic_dimension))
        states_raw = np.load(states_dir)
        states = np.concatenate([states_raw[ids, :cfg.num_frames-1, :1], states_raw[ids, 1:, :1]], axis=2)
        states = np.reshape(states, (-1, cfg.intrinsic_dimension))
        latent = np.load(os.path.join(latent_dir, 'refine_latent.npy'))
        return (states, latent)
    
    eval_fit('train')
    eval_fit('val')
    eval_fit('test')
    pprint.pprint(all_loss)

if __name__ == '__main__':
    if str(sys.argv[4]) == 'eval-train':
        gather_latent_from_trained_high_dim_model()
    elif str(sys.argv[4]) == 'eval-refine-train':
        gather_latent_from_trained_refine_model()
    elif str(sys.argv[4]) == 'eval-sr-train':
        eval_sr_model()
    else:
        main()