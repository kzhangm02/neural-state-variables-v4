import os
import sys
import yaml
import glob
import torch
import pprint
from munch import munchify
from models import VisDynamicsModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

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
    high_dim_ckpt_dir = str(sys.argv[2])
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
        if_test=False,
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
        high_dim_ckpt_dir=high_dim_ckpt_dir
    )

    # define callback for selecting checkpoints during training
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir + '/lightning_logs/checkpoints/',
        filename='epoch={epoch}-val_loss={val/loss:.4f}',
        auto_insert_metric_name=False,
        verbose=True,
        monitor='val/loss',
        mode='min',
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir + '/lightning_logs/last_checkpoints/',
        filename='epoch={epoch}-val_loss={val/loss:.4f}',
        auto_insert_metric_name=False,
    )

    # define trainer
    # if cfg.seed == 1:
    #     path = '/home/kevin/neural-state-variables-v4/logs/elastic_pendulum_encoder-decoder-64_1/lightning_logs/last_checkpoints/epoch=49-val_loss=62.3261.ckpt'
    # if cfg.seed == 2:
    #     path = '/home/kevin/neural-state-variables-v4/scripts/logs_double_pendulum_encoder-decoder-64_2/lightning_logs/last_checkpoints/epoch=999-val_loss=107.0373.ckpt'
    # if cfg.seed == 3:
    #     path = '/home/kevin/neural-state-variables-v4/scripts/logs_double_pendulum_encoder-decoder-64_3/lightning_logs/last_checkpoints/epoch=999-val_loss=64.7364.ckpt'

    trainer = Trainer(
        accelerator='gpu',
        num_nodes=cfg.num_gpus,
        max_epochs=cfg.epochs,
        deterministic=True,
        strategy='ddp',
        # strategy='ddp_find_unused_parameters_true',
        default_root_dir=log_dir,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_checkpointing=True,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
    )

    trainer.fit(model)
    # trainer.fit(model, ckpt_path=path)

if __name__ == '__main__':
    main()
