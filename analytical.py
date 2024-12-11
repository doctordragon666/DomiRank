# 执行分析解方法

import drank as dr
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import time

N = 2500  # 网络大小
m = 3  # 每个节点的平均连接数量
directed = False

##### #RANDOMIZATION ######

# 随机结果
# seed = np.random.randint(0, high = 2**32-1)
# 固定结果
seed = 42

# 设置随机种子
np.random.seed(seed)


############## 注意：在这里你可以注释来创建你想要的任何图表，作者创建的是E-R图 ############
G = nx.fast_gnp_random_graph(N, 2*m/N, directed=directed, seed=seed)

# 在这里和下面插入网络图 ########################3
GAdj = nx.to_scipy_sparse_array(G)
# 如果它是有向的（取决于链接的交互方式...）则翻转整个网络的方向
if directed:
    GAdj = sp.sparse.csr_array(GAdj.T)
G, node_map = dr.relabel_nodes(G, yield_map=True)
print(type(GAdj))
# 这里我们使用domirank算法找到最大特征值，使用黄金分割/二分法搜索空间，利用sigma > -1/lambN时的快速发散
# t1 = time.time()
# 有时候你需要改变这些参数来收敛
lambN = dr.find_eigenvalue(GAdj, maxIter=500, dt=0.01, checkStep=25)
# t2 = time.time()
# 重要提示：如果您不想比较domirank特征值计算与numpy计算的速度，请注释时间函数time的调用，但是对于特别大的网络请取消掉时间函数的调用

print(f'\nThe found smallest eigenvalue was: lambda_N = {lambN}')
# print(f'\nOur single-threaded algorithm took: {t2-t1}s')

# 注意，如果你只是执行dr.domirank(GAdj)并且没有传递最优的sigma，它将找到它自己。
# 使用之前空间（0，-1/lambN）计算获取最佳sigma
sigma, sigmaArray = dr.optimal_sigma(GAdj, analytical=True, endVal=lambN)
print(f'\n The optimal sigma was found to be: {sigma*-lambN}/-lambda_N')


fig1 = plt.figure(1)
ourRange = np.linspace(0, 1, sigmaArray.shape[0])
index = np.where(sigmaArray == sigmaArray.min())[0][-1]

plt.plot(ourRange, sigmaArray)
plt.plot(ourRange[index], sigmaArray[index], 'ro', mfc='none', markersize=10)
plt.xlabel('sigma')
plt.ylabel('loss')
plt.savefig("sigma.png")


# generate the centrality using the optimal sigma
analyticalDomiRankDistribution = dr.domirank_by_annalytical(
    GAdj, sigma=sigma)
# generate the attack using the centrality (descending)
analyticalDomiRankAttack = dr.generate_attack(analyticalDomiRankDistribution)
# attack the network and get the largest connected component evolution
domiRankRobustnessA, domiRankLinksA = dr.network_attack_sampled(
    GAdj, analyticalDomiRankAttack)

# generating the plot
fig2 = plt.figure(2)
ourRangeNew = np.linspace(0, 1, domiRankRobustnessA.shape[0])
plt.plot(ourRangeNew, domiRankRobustnessA, label='Analytical DR')
plt.legend()
plt.xlabel('fraction of nodes removed')
plt.ylabel('largest connected component')
plt.savefig("resulta.png")
