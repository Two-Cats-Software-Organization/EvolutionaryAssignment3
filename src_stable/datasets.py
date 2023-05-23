import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from functools import lru_cache

class NParity(torch.utils.data.Dataset):
    """Some Information about NParity"""
    def __init__(self, N, max_set_size=2**31):
        super().__init__()
        self.N = N
        self.set_size = 2**N
        self.max_set_size = max_set_size
        self.overflow = self.set_size > self.max_set_size
        

    @lru_cache()
    def __getitem__(self, index):
        if self.overflow:
            return self._getitem_overflow()
        # x = torch.Tensor([int(q) for q in bin(index)[-self.N:]]).to(torch.int8)
        x = torch.Tensor(list(map(int, format(index, 'b').zfill(self.N)))).to(torch.int8)
        # x = torch.zeros(self.N, dtype=torch.int8)
        y = x.sum() % 2 # 奇数个1时，校验位为1，当有偶数个1时校验位为0。 补全为偶数个1。
        return x, y

    def _getitem_overflow(self):
        x = torch.randint(0, 2, (self.N,), dtype=torch.int8)
        y = x.sum() % 2
        return x, y
        
    def __len__(self):
        return min(self.set_size, self.max_set_size)
    
if __name__ == "__main__":
    problem = NParity(1000)
    loader = torch.utils.data.DataLoader(problem, batch_size=4, shuffle=False)
    for i, (x, y) in enumerate(loader):
        print(x, y)
        if i>100:
            break
    