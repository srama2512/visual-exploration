import random
import numpy as np

class BaseDataLoader(object):
    """
    Base DataLoader class for abstracting the reading, batching and shuffling operations
    """

    def __init__(self, opts):
        """
        Loads the dataset and saves settings needed
        """
        # ---- Load the dataset ----
        self._load_data(opts)

        # ---- Save settings needed for batching operations ----
        # Dataset statistics
        self.counts = {split: self.data[split].shape[0] for split in self.data.keys()}

        # Iteration tracker
        self.iter_idxes = {split: 0 for split in self.data.keys()}


        # Shuffle the training data indices and access them in the shuffled order
        self.shuffle = opts.shuffle
        self.shuffled_idx = list(range(self.data['train'].shape[0]))
        if self.shuffle:
            random.shuffle(self.shuffled_idx)
        self.shuffled_idx = np.array(self.shuffled_idx)

        # Other statistics
        self.pano_shape = self.data['train'].shape[1:]
        self.batch_size = opts.batch_size
        self.N, self.M = self.data['train'].shape[1:3]

    def next_batch(self, split):
        """
        Returns the next batch from split
        depleted: is the epoch over?
        """
        # Get indices of data
        batch_size = min(self.batch_size, self.counts[split] - self.iter_idxes[split])
        start_idx = self.iter_idxes[split]
        end_idx = start_idx + batch_size
        batch_idxes = self.shuffled_idx[range(start_idx, end_idx)] if split == 'train' else range(start_idx, end_idx)

        # Get batch of data
        outputs = self._get_data(split, batch_idxes)

        # Update iterators
        if self.iter_idxes[split] + batch_size == self.counts[split]:
            depleted = True
            self.iter_idxes[split] = 0
        else:
            depleted = False
            self.iter_idxes[split] += batch_size

        return outputs + (depleted, )

    def _load_data(self, opts):
        """
        Loads the required data into self.data, self.labels (optional), etc
        """
        pass

    def _get_data(self, split, idxes):
        """
        Given a split and idxes (some iterator), return the images and labels (optional), etc
        """
        pass
