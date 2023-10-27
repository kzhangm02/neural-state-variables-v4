import os
import yaml
import pprint
import numpy as np
from munch import munchify
from dso import DeepSymbolicRegressor

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def main():
    # np.random.seed(0)
    # X = np.random.random((10, 2))
    # y = np.sin(X[:,0]) + X[:,1] ** 2

    # # Create the model
    # print(X.shape, y.shape)
    # model = DeepSymbolicRegressor('./dso_config.json') # Alternatively, you can pass in your own config JSON path

    # # Fit the model
    # model.fit(X, y) # Should solve in ~10 seconds

    config_filepath = '../configs/single_pendulum/regression/config1.yaml'
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)

    log_dir = '_'.join([cfg.log_dir, cfg.dataset, cfg.model_name.replace('sr', '64'), str(cfg.seed)])
    variable_dir = os.path.join(log_dir, 'variables_train')
    ids = np.load(os.path.join(variable_dir, 'ids.npy'))
    ids = [int(id[:id.index('_')]) for id in ids][::cfg.num_frames-1]

    states_dir = os.path.join(cfg.data_filepath, 'collect', cfg.dataset, 'states.npy')
    states = np.load(states_dir)
    states = states[ids, :cfg.num_frames-1]
    states = np.reshape(states, (-1, cfg.intrinsic_dimension))
    latent = np.load(os.path.join(variable_dir, 'refine_latent.npy'))
    latent = latent[:, 0]

    model = DeepSymbolicRegressor('./dso_config.json')

    model.fit(states, latent)

    print(model.program_.pretty())

if __name__ == '__main__':
    main()