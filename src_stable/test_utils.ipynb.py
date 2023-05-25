#%%
from utils import *
#%%
t = torch.tensor([1, 2, 3], dtype=torch.float32)
# list(gpu2numpy(t)) # 无病detach也是可以的
gpu2numpy(t) # 无病detach也是可以的
gpu2numpy(t, t) # 无病detach也是可以的
t, t = numpy2gpu(np.array([1, 2, 3]), np.array([1, 2, 3]))
t
# t = torch.tensor([1, 2, 3], dtype=torch.float32, device='cuda')
# gpu2numpy(t)
# t = torch.tensor([1, 2, 3], dtype=torch.float32, device='cuda', requires_grad=True)
# gpu2numpy(t)
#%%