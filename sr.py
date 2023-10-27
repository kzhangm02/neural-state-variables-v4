import os
import sys
import csv
import yaml
import sympy
import pprint
import shutil
import numpy as np
from sr_utils import *
from munch import munchify
from pysr import PySRRegressor

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def cleanup(cfg, sr_dir):

    data = [['Variable', 'Loss', 'Equation']]
    for k in range(cfg.intrinsic_dimension):
        fit_filepath = os.path.join(sr_dir, f'model.csv.out{k+1}')
        fit = np.genfromtxt(fit_filepath, delimiter=',', dtype=None, encoding='utf-8')
        loss, eq = fit[1:, 1].astype(float), fit[1:, 2]
        idx = np.argmin(loss)
        data.append([k+1, loss[idx], simplify(eq[idx], 1e-6)])
        
    with open(os.path.join(sr_dir, 'fit.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    for k in range(cfg.intrinsic_dimension):
        fit_filepath = os.path.join(sr_dir, f'model.csv.out{k+1}')
        os.remove(f'{fit_filepath}.bkup')
        os.remove(fit_filepath)

def main():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)

    log_dir = '_'.join([cfg.log_dir, cfg.dataset, cfg.model_name.replace('sr', '64'), str(cfg.seed)])
    variable_dir = os.path.join(log_dir, 'variables_train')
    ids = np.load(os.path.join(variable_dir, 'ids.npy'))
    ids = [int(id[:id.index('_')]) for id in ids][::cfg.num_frames-1]

    states_dir = os.path.join(cfg.data_filepath, 'collect', cfg.dataset, 'states.npy')
    states_raw = np.load(states_dir)
    states = np.concatenate([states_raw[ids, :cfg.num_frames-1, :1], states_raw[ids, 1:, :1]], axis=2)
    states = np.reshape(states, (-1, cfg.intrinsic_dimension))
    latent = np.load(os.path.join(variable_dir, 'refine_latent.npy'))

    sr_dir = os.path.join(log_dir, 'symbolic_regression')
    if os.path.exists(sr_dir):
        shutil.rmtree(sr_dir)
    os.makedirs(sr_dir)

    model = PySRRegressor(
        niterations=cfg.niterations,
        maxsize=cfg.maxsize,
        loss="L1DistLoss()",
        batching=True,
        batch_size=cfg.batch_size,
        model_selection='accuracy',
        complexity_of_constants=cfg.complexity_of_constants,
        # weight_add_node=cfg.weight_add_node,
        # weight_insert_node=cfg.weight_insert_node,
        # weight_delete_node=cfg.weight_delete_node,
        # weight_do_nothing=cfg.weight_do_nothing,
        # weight_mutate_constant=cfg.weight_mutate_constant,
        # weight_mutate_operator=cfg.weight_mutate_operator,
        # weight_simplify=cfg.weight_simplify,
        # weight_randomize=cfg.weight_randomize,
        binary_operators=cfg.binary_operators,
        unary_operators=cfg.unary_operators,
        constraints=cfg.constraints,
        nested_constraints=cfg.nested_constraints,
        equation_file=os.path.join(sr_dir, 'model.csv'),
        random_state=cfg.seed,
        deterministic=True,
        procs=0,
    )

    model.fit(states, latent)

    cleanup(cfg, sr_dir)

if __name__ == '__main__':
    main()