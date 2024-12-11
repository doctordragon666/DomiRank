import drank as dr
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import time

N = 2500  # size of network网络大小
m = 3  # average number of links per node.每个节点的平均连接数量
# if you want to use the analytical method or the recursive definition如果你想要使用分析方法或者递归定义
analytical = False
directed = False

##### #RANDOMIZATION ######

# for random results随机结果
seed = np.random.randint(0, high=2**32-1)

# for deterministic results固定结果
# seed = 42

# setting the random seed
np.random.seed(seed)

##### END OF RANDOMIZATION #####

############## 注意：在这里你可以注释来创建你想要的任何图表，作者创建的是E-R图 ############
############## IMPORTANT!!!! Here you can create whatever graph you want and just comment this erdos-renyi network out ############
# THIS IS THE INPUT, CHANGE THIS TO ANY GRAPH #######
G = nx.fast_gnp_random_graph(N, 2*m/N, directed=directed, seed=seed)
nx.draw_networkx(G)
plt.savefig('graph.png')

# 在这里和下面插入图 ########################3
# insert network hereunder ########################3
GAdj = nx.to_scipy_sparse_array(G)
# 转置网络来改变方向
# flipping the network direction if it is directed (depends on the interactions of the links...)
if directed:
    GAdj = sp.sparse.csr_array(GAdj.T)
G, node_map = dr.relabel_nodes(G, yield_map=True)
print(type(GAdj))
# 这里我们通过domirank算法找到最大的特征值，并且通过黄金分割和二分算法来搜索空间，当sigma > -1/lambN快速发散
# Here we find the maximum eigenvalue using the DomiRank algorithm and searching the space through a golden-ratio/bisection algorithm, taking advantage of the fast divergence when sigma > -1/lambN
t1 = time.time()
# sometimes you will need to change these parameters to get convergence有时候你需要改变这些参数来收敛
lambN = dr.find_eigenvalue(GAdj, maxIter=500, dt=0.01, checkStep=25)
t2 = time.time()
# 重要提示：如果您不想比较domirank特征值计算与numpy计算的速度，请注释时间函数time的调用，但是对于特别大的网络请取消掉时间函数的调用
# IMPORTANT NOTE: for large graphs, comment out the lines below (23-26), along with lines (32-33).
# Please comment the part below (23-26) & (32-33) if you don't want a comparison with how fast the domirank eigenvalue computation is to the numpy computation.

print(f'\nThe found smallest eigenvalue was: lambda_N = {lambN}')
print(f'\nOur single-threaded algorithm took: {t2-t1}s')

# 注意，如果你只是执行dr. domirank（GAdj）并且没有通过最佳sigma，它会自己找到它。
# note, if you just perform dr.domirank(GAdj) and dont pass the optimal sigma, it will find it itself.
# get the optimal sigma using the space (0, -1/lambN) as computed previously 使用之前计算的空间（0，-1/lambN）获取最佳sigma
sigma, sigmaArray = dr.optimal_sigma(GAdj, analytical=analytical, endVal=lambN)
print(f'\n The optimal sigma was found to be: {sigma*-lambN}/-lambda_N')


fig1 = plt.figure(1)
ourRange = np.linspace(0, 1, sigmaArray.shape[0])
index = np.where(sigmaArray == sigmaArray.min())[0][-1]

plt.plot(ourRange, sigmaArray)
plt.plot(ourRange[index], sigmaArray[index], 'ro', mfc='none', markersize=10)
plt.xlabel('sigma')
plt.ylabel('loss')


# generate the centrality using the optimal sigma
_, ourDomiRankDistribution = dr.domirank_by_recursive(
    GAdj, sigma=sigma)
# generate the attack using the centrality (descending)
ourDomiRankAttack = dr.generate_attack(ourDomiRankDistribution)
# attack the network and get the largest connected component evolution
domiRankRobustness, domiRankLinks = dr.network_attack_sampled(
    GAdj, ourDomiRankAttack)

# generating the plot
fig2 = plt.figure(2)
ourRangeNew = np.linspace(0, 1, domiRankRobustness.shape[0])
plt.plot(ourRangeNew, domiRankRobustness, label='Recursive DR')
plt.legend()
plt.xlabel('fraction of nodes removed')
plt.ylabel('largest connected component')

# plt.show()
# in linux use png storage
plt.savefig("result.png")
