import numpy as np
from scipy import stats
import skdim
import os
import sys
import yaml
from munch import munchify

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)


def calculate_intrinsic_dimension_statistics(dataset):
    configs_dir = os.path.join('../configs', dataset, 'model64')
    filename = 'mu.npy'
    dims_all = []
    tle = skdim.id.DANCo(k=10, fractal=False)
    for config_filepath in os.listdir(configs_dir):
        cfg = load_config(filepath=os.path.join(configs_dir, config_filepath))
        cfg = munchify(cfg)
        if cfg.seed == 1:
            log_dir_name = '_'.join([cfg.dataset, cfg.model_name, str(cfg.seed)])
            log_dir = os.path.join(cfg.log_dir, log_dir_name)
            vars_filepath = os.path.join(log_dir, 'variables')
            vars = np.load(os.path.join(vars_filepath, filename))
            vars = np.unique(vars, axis=0)
            tle_dim = tle.fit(vars).dimension_
            dims_all.append(tle_dim)
    dim_mean = np.mean(dims_all)
    dim_std = np.std(dims_all)
    print('Mean (±std):', '%.2f (±%.2f)' % (dim_mean, dim_std))
    print('Confidence interval:', '(%.1f, %.1f)' % (dim_mean-1.96*dim_std, dim_mean+1.96*dim_std))

if __name__ == '__main__':
    dataset = str(sys.argv[1])
    calculate_intrinsic_dimension_statistics(dataset)