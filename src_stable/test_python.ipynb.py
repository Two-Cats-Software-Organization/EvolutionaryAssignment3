#%%
import bisect
array = [1, 2, 3, 4, 5]
# bisect.bisect(array, 3)
# bisect.bisect(array, 5)
# bisect.bisect(array, 2)
# bisect.bisect(array, 1)
bisect.bisect(array, 0.5)
bisect.bisect(array, 1.5)
# bisect.bisect(array, 5.5)
#%%
import tqdm
epochs = 1000
bar = tqdm.tqdm(range(epochs))
# for i in bar:
#     def try_continue():
#         print(f"Hello {i}")
#         continue
#     try_continue()
#     print(f"World {i}")
#%%
import time
for i in tqdm.tqdm(range(epochs), desc="外层", position=0, leave=True, colour='green'
                #    ncols=100
                   ):
    for j in tqdm.tqdm(range(10), desc="内层", position=1, leave=False, colour='yellow'
                    #    ncols=100
                       ):
        # time.sleep(0.1)
        pass
for j in tqdm.tqdm(range(10), desc="内层", position=1, leave=False, colour='yellow'):
    time.sleep(0.5)
for j in tqdm.tqdm(range(10), desc="内层", position=1, leave=False, colour='yellow'):
    time.sleep(0.5)
exit()
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class MyModule(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super(MyModule, self).__init__()
        self.a = 1
        self.b = nn.Parameter(torch.zeros(5), requires_grad=False)

    def forward(self, x):

        return x    

m = MyModule()
d = m.state_dict()   
d
#%%
m.b[0] = 100
m.state_dict() 
d # 没有保存状态的作用。实际上改变了m.b的值，d也会改变。
# %%
from copy import deepcopy
m = MyModule()
m.__dict__
#%%
old_obj = deepcopy(m)
m.b[0] = 100    
m.b, old_obj.b
# %%
m.__dict__.update(old_obj.__dict__)
m.b, old_obj.b
# %%
