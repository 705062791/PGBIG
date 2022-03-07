from torch.utils.data import Dataset
import numpy as np
from utils import data_utils


class CMU_Motion3D(Dataset):

    def __init__(self, opt, split, actions='all'):

        self.path_to_data = opt.data_dir
        input_n = opt.input_n
        output_n = opt.output_n

        self.split = split
        is_all = actions
        actions = data_utils.define_actions_cmu(actions)
        # actions = ['walking']
        if split == 0:
            path_to_data = self.path_to_data + '/train/'
            is_test = False
        else:
            path_to_data = self.path_to_data + '/test/'
            is_test = True


        if not is_test:
            all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d_all(opt, path_to_data, actions,
                                                                                             input_n, output_n,
                                                                                             is_test=is_test)
        else:
            # all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d_all(opt, path_to_data, actions,
            #                                                                 input_n, output_n,
            #                                                                 is_test=is_test)

            all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d_n(opt, path_to_data, actions,
                                                                                             input_n, output_n,
                                                                                             is_test=is_test)

        self.all_seqs = all_seqs
        self.dim_used = dim_use

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        return self.all_seqs[item]
