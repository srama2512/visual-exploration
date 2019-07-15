import h5py
import torch
import numpy as np

from base.base_data_loader import BaseDataLoader

def prepro_metric(x, metric_type):
    if metric_type == 'surface_area':
        # x - N array
        return np.clip(x / 25000.0, 0.0, 1.0)
    elif metric_type == 'relative_volume':
        # x = N array
        return np.clip(x / 0.6, 0.0, 1.0)
    else:
        return x

class DataLoaderSimple(BaseDataLoader):
    """
    Simple DataLoader without expert rewards or trajectories.
    """

    def __init__(self, opts):
        super(DataLoaderSimple, self).__init__(opts)

    def _load_data(self, opts):
        # Load images
        h5_file = h5py.File(opts.h5_path, 'r')
        self.data = {split: np.array(h5_file[split]) for split in ['train', 'val', 'test']}
        h5_file.close()

        # Load labels
        labels_file = h5py.File(opts.labels_path, 'r')
        self.labels = {split: np.array(labels_file[opts.metric_type][split])
                                                          for split in ['train', 'val', 'test']}
        self.labels = {split: prepro_metric(labs, opts.metric_type)
                                                          for split, labs in self.labels.iteritems()}

        labels_file.close()

    def _get_data(self, split, idxes):
        out_imgs = np.copy(self.data[split][idxes])
        out_labs = np.copy(self.labels[split][idxes])
        return out_imgs, out_labs

class DataLoaderExpertPolicy(DataLoaderSimple):
    """
    DataLoader that additionally returns expert trajectories
    """

    def __init__(self, opts):
        super(DataLoaderExpertPolicy).__init__(opts)

    def _load_data(self, opts):
        # ---- Load the images, labels ----
        super(DataLoaderExpertPolicy, self)._load_data(opts)

        # ---- Load the expert trajectories ----
        self.trajectories_type = opts.trajectories_type
        if opts.trajectories_type == 'utility_maps':
            # Load the utility maps
            utility_file = h5py.File(opts.utility_h5_path)
            self.utility_maps = {}
            # These are KxNxMxNxM arrays
            for split in utility_file.keys():
                self.utility_maps[split] = np.array(utility_file[split]['utility_maps'])
        elif opts.trajectories_type == 'expert_trajectories':
            # Load the trajectories
            # {split: #split_samples x T-1 numpy array}
            self.trajectories = torch.load(opts.utility_h5_path)
        elif opts.trajectories_type == 'saliency_scores':
            # Load the saliency scores
            h5_file = h5py.File(opts.utility_h5_path)
            self.saliency_scores = {}
            # These are MxNxB arrays (transposed due to matlab)
            for split in h5_file.keys():
                split_data = np.array(h5_file[split])
                ndim = len(split_data)
                self.saliency_scores[split] = split_data.transpose(*reversed(range(ndim)))
        else:
            raise ValueError('Wrong trajectories_type!')

    def _get_data(self, split, idxes):
        out_imgs = np.copy(self.data[split][idxes])
        out_labs = np.copy(self.labels[split][idxes])
        if self.trajectories_type == 'utility_maps':
            out_maps = self.utility_maps[split][idxes]
        elif self.trajectories_type == 'expert_trajectories':
            out_maps = {}
            out_maps = {self.trajectories[split][(i, j)][idxes] for i in range(0, self.N)
                                                                for j in range(self.M)}
        elif self.trajectories_type == 'saliency_scores':
            out_maps = self.saliency_scores[split][idxes]

        return out_imgs, out_labs, out_maps
