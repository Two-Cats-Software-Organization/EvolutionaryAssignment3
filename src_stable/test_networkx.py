#%%
import networkx as nx
G = nx.DiGraph()
# G.add_nodes_from([(0, {"name":"input_1"}),1,2])
# G.add_weighted_edges_from([(0,2,0.25),(1,2, 0.5)])
G.add_weighted_edges_from([("input1","output1",0.25),("input2","output1", 0.5)])
G.nodes["input1"]['label'] = "My Label1" # 只对latex有效
G.nodes["input2"]['label'] = "My Label2" # 只对latex有效
G.nodes["output1"]['label'] = "My Label3" # 只对latex有效
# G[1]
# %%
# VSCode 问题 https://blog.csdn.net/weixin_44228113/article/details/129355881
from matplotlib import pyplot as plt
plt.style.use('default')
# plt.style.use('seaborn-whitegrid')
# nx.draw(G, with_labels=True,)
nx.drawing.draw_networkx(G, with_labels=True, 
                         labels={n:G.nodes[n]['label'] for n in G.nodes},)
plt.savefig("fig/path.png")
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
a = torch.Tensor([0, 0, 1, 0, 1]).to("cuda")
a = torch.Tensor([0, 0, 0, 0, 0])
b = a.nonzero()
list(b)
# %%
from individual import Individual
indi = Individual(input_dim=2, output_dim=2, max_hidden_dim=3)
#%%
import networkx as nx
g = indi.to_networkx()
g.nodes, g.edges
#%%
indi.node_existence
indi.connectivity
#%%
# {n:G.nodes[n]['label'] for n in G.nodes}
#%%
nx.drawing.draw_networkx(g, with_labels=True, 
                         labels={n:g.nodes[n]['label'] for n in g.nodes},
                         )
plt.savefig("fig/indi.png")
# %%
p =  nx.nx_pydot.to_pydot(g)
p
# %%
nx.drawing.nx_pydot.write_dot(g, "fig/indi.dot")

# %%
