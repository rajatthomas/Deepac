from torch.utils.data import Dataset
import os.path as osp
import numpy as np
import torch

class PAC_data(Dataset):

    def __init__(self, opt, split, transform=None):
        """

        :param opt: Command line option(/defaults)
        :param split: train | val | test
        :param transform: NotImplemented
        """
        data_file = osp.join(opt.root_dir, opt.data_file)
        all_data = np.load(data_file)

        data = all_data[split]

        if opt.standardize:

            mask = data['mask_2d'].reshape(data.shape[1:])

            n_subj = data.shape[0]
            for i_subj in range(n_subj):
                data_subj = data[i_subj]
                mean_subj = data_subj[mask, :].mean(axis=0)
                std_subj = data_subj[mask, :].std(axis=0)
                if np.any(std_subj == 0) or np.any(np.isnan(mean_subj)) or np.any(np.isnan(std_subj)):
                    import pdb;
                    pdb.set_trace()
                data[i_subj] = mask[..., np.newaxis] * (data_subj - mean_subj) / std_subj

        self.data = torch.from_numpy(data)

        if split == 'train_3d':
            y_split = 'y_train'
        if split == 'valid_3d':
            y_split = 'y_valid'
        if split == 'test_3d':
            y_split = 'y_test'

        self.labels = torch.from_numpy(all_data[y_split])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


def get_data_set(opt, split, transform=None):

    data_set = PAC_data(opt,
                        split=split,
                        transform=transform)
    return data_set

