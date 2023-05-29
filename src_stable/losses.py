import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
Tensor = torch.Tensor
def prechelt_mse_loss(input: Tensor, target: Tensor):
    # 注意input是y_pred, target是y_true
    if input.shape[0]==1:
        return F.mse_loss(input, target)
    return F.mse_loss(input, target)*(input.max()-input.min())*100
    