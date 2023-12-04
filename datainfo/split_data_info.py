import os
import json
import numpy as np

seed = 3
dataset = 'swing_stick'

data_dir = f'/data/physics_prediction_v2/data/{dataset}'
num_videos = len(os.listdir(data_dir))
idxs = list(range(num_videos))

np.random.seed(seed)
np.random.shuffle(idxs)
train_split = int(0.8 * num_videos)
val_split = int(0.1 * num_videos)
train_idxs = idxs[:train_split]
val_idxs = idxs[train_split:train_split + val_split]
test_idxs = idxs[train_split + val_split:]
print('Train videos: ', len(train_idxs))
print('Valid videos: ', len(val_idxs))
print('Test videos:  ', len(test_idxs))

data_split_dict = {
    "test": test_idxs,
    "val": val_idxs,
    "train": train_idxs,
}

with open(f'./{dataset}/data_split_dict_{seed}.json', 'w') as f:
    json.dump(data_split_dict, f, indent=4)
