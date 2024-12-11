import drank as dr
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 生成一个复杂网络
G = nx.erdos_renyi_graph(100, 0.1)
fig = plt.figure(1)
layout = nx.random_layout(G)
nx.draw_networkx(G, pos=layout)
plt.savefig('graph_ER.png')
GAdj = nx.to_scipy_sparse_array(G)

# 生成复杂网络的domirank，这里不使用分析方法，而是使用递归算法
lambN = dr.find_eigenvalue(GAdj, maxIter=500, dt=0.01, checkStep=25)
sigma, _ = dr.optimal_sigma(GAdj, analytical=False, endVal=lambN)
print(f'\n The optimal sigma was found to be: {sigma*-lambN}/-lambda_N')
_, domirank = dr.domirank_by_recursive(G, sigma=sigma)
# sigma, _ = dr.optimal_sigma(GAdj, analytical=True, endVal=lambN)
# print(f'\n The optimal sigma was found to be: {sigma*-lambN}/-lambda_N')
# domirank = dr.domirank_by_annalytical(G, sigma=sigma)


# 生成根据domirank分布的攻击结果和普通rank的攻击结果
domiRankAttack = dr.generate_attack(domirank)
rankAttack = dr.generate_attack(nx.degree_centrality(G))
domiRankRobustness, domiRankLinks = dr.network_attack_sampled(
    GAdj, domiRankAttack)
rankRobustness, rankLinks = dr.network_attack_sampled(
    GAdj, rankAttack)

fig = plt.figure(2)
ourRangeNew = np.linspace(0, 1, domiRankRobustness.shape[0])
plt.plot(ourRangeNew, domiRankRobustness, label='Recursive DR')
plt.legend()
plt.xlabel('fraction of nodes removed')
plt.ylabel('largest connected component')
plt.savefig("result_domiattack.png")

fig = plt.figure(3)
ourRangeNew = np.linspace(0, 1, rankRobustness.shape[0])
plt.plot(ourRangeNew, rankRobustness, label='Recursive DR')
plt.legend()
plt.xlabel('fraction of nodes removed')
plt.ylabel('largest connected component')
plt.savefig("result_attack.png")
