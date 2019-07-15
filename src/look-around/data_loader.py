import pdb
import h5py
import torch
import numpy as np

from base.base_data_loader import BaseDataLoader

class DataLoaderSimple(BaseDataLoader):
    """
    Simple DataLoader without expert rewards or trajectories.
    """

    def __init__(self, opts):
        super(DataLoaderSimple, self).__init__(opts)
        # Dataset statistics for unseen classes
        if opts.h5_path_unseen != '':
            self.test_unseen_count = self.data['test_unseen'].shape[0]
            self.test_unseen_idx = 0

    def _load_data(self, opts):
        # ---- Load the dataset ----
        h5_file = h5py.File(opts.h5_path, 'r')
        self.data = {split: np.array(h5_file[split]) for split in ['train', 'val', 'test']}
        h5_file.close()
        # ---- Load the unseen classes (if available)----
        if opts.h5_path_unseen != '':
            h5_file_unseen = h5py.File(opts.h5_path_unseen, 'r')
            self.data['test_unseen'] = np.array(h5_file_unseen['test'])
            h5_file_unseen.close()

    def _get_data(self, split, idxes):
        out_imgs = np.copy(self.data[split][idxes])
        return (out_imgs, )

class DataLoaderExpert(DataLoaderSimple):
    """
    DataLoader that additionally returns expert rewards
    """
    def __init__(self, opts):
        super(DataLoaderExpert, self).__init__(opts)

    def _load_data(self, opts):
        # ---- Loads images ----
        super(DataLoaderExpert, self)._load_data(opts)
        # ---- Load rewards ----
        rewards_file = h5py.File(opts.rewards_h5_path)
        self.rewards = {}
        # These are KxNxM arrays containing rewards corresponding to each views of
        # all panoramas in the train and val splits
        self.rewards['train'] = np.array(rewards_file['train/nms'])
        self.rewards['val'] = np.array(rewards_file['val/nms'])

    def _get_data(self, split, idxes):
        out_imgs = np.copy(self.data[split][idxes])
        # test, test_unseen data do not have expert rewards
        out_rewards = self.rewards[split][idxes] if split in self.rewards else None
        return out_imgs, out_rewards

class DataLoaderExpertPolicy(DataLoaderSimple):
    """
    DataLoader that additionally returns expert trajectories
    """

    def __init__(self, opts):
        super(DataLoaderExpertPolicy, self).__init__(opts)

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
        if self.trajectories_type == 'utility_maps':
            out_maps = self.utility_maps[split][idxes]
        elif self.trajectories_type == 'expert_trajectories':
            out_maps = {}
            out_maps = {self.trajectories[split][(i, j)][idxes] for i in range(self.N)
                                                                for j in range(self.M)}
        elif self.trajectories_type == 'saliency_scores':
            out_maps = self.saliency_scores[split][idxes]

        return out_imgs, out_maps
