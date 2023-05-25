#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
import torch.utils.data

import numpy as np
from typing import Callable, List, Tuple, Iterable


def gpu2numpy(*tensors: torch.Tensor)->np.ndarray | Tuple[np.ndarray]:
    assert len(tensors) > 0
    if len(tensors) == 1:
        return tensors[0].detach().cpu().numpy()
    return tuple(t.detach().cpu().numpy() for t in tensors)
    # return tensor.detach().cpu().numpy(), *tuple(t.detach().cpu().numpy() for t in tensors)
# def gpu2numpy(tensors: torch.Tensor|Iterable[torch.Tensor]):
#     match tensors:
#         case torch.Tensor:
#             return tensors.detach().cpu().numpy()
#         case Iterable[torch.Tensor]:
#             return tuple(t.detach().cpu().numpy() for t in tensors)
#         case _:
#             raise TypeError(f'Expected torch.Tensor or Collection[torch.Tensor], got {type(tensors)}')
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
def numpy2gpu(*arrays: np.ndarray, device=device)->torch.Tensor| Tuple[torch.Tensor]:
    assert len(arrays) > 0
    if len(arrays) == 1:
        return torch.tensor(arrays[0], device=device).to(torch.float32)
    return tuple(torch.tensor(a, device=device).to(torch.float32) for a in arrays)
#%%


class EarlyStopper(object):

    def __init__(self, patience=10, min_delta=0, cumulative_delta=False):
        """Early Stopper

        Args:
            patience (int): 多少轮没有改进就没有耐心了
            min_delta (float): 多少的改进才叫改进
            cumulative_delta (bool): 每一步的改进很小不算你改进，但是也是你未来改进评价的障碍。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta

        self.patience_counter = 0
        self.best_score = -torch.inf
        self.best_epoch = -1
        self.best_state_dict = None

    def is_continuable(self, epoch_i, score, state_dict=None):
        if score <= self.best_score+self.min_delta:
            # 没有改进
            if not self.cumulative_delta and score > self.best_score:
                # 有微小的改进，于是更新best_score，这次改进也算你。
                self.best_score = score
                self.best_state_dict = state_dict
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return False
            return True
        else:
            # 有所改进
            self.best_score = score
            self.best_state_dict = state_dict
            self.best_epoch = epoch_i
            self.patience_counter = 0
            return True


def dataset2numpy(dataset: torch.utils.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """将torch.utils.data.Dataset转换为两个X和Y两个numpy数组

    Args:
        dataset (torch.utils.data.Dataset): 输入数据集

    Returns:
        Tuple[np.ndarray, np.ndarray]: X和Y两个numpy数组。
    """
    probe_X = dataset[0][0].detach().cpu().numpy()
    probe_Y = dataset[0][1].detach().cpu().numpy()
    
    X = np.zeros((len(dataset), *probe_X.shape), dtype=probe_X.dtype)
    Y = np.zeros((len(dataset), *probe_Y.shape), dtype=probe_Y.dtype)
    for i in range(len(dataset)):
        x, y = dataset[i]
        X[i] = gpu2numpy(x)
        Y[i] = gpu2numpy(y)
    return X, Y


#%%

