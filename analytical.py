# 执行分析解方法

import domirank as dr
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import time

N = 2500 #网络大小
m = 3 #每个节点的平均连接数量
analytical = False #如果你想要使用分析方法或者递归定义，定义为true
directed = False

##### #RANDOMIZATION ######

#for random results随机结果
# seed = np.random.randint(0, high = 2**32-1)

#for deterministic results固定结果
seed = 42

#setting the random seed
np.random.seed(seed)


##### END OF RANDOMIZATION #####

############## 注意：在这里你可以注释来创建你想要的任何图表，作者创建的是E-R图 ############
G = nx.fast_gnp_random_graph(N, 2*m/N, directed = directed, seed = seed) #####THIS IS THE INPUT, CHANGE THIS TO ANY GRAPH #######

#################### 在这里和下面插入网络图 ########################3
#################### insert network hereunder ########################3
GAdj = nx.to_scipy_sparse_array(G)
#flipping the network direction if it is directed (depends on the interactions of the links...)翻译：如果它是有向的（取决于链接的交互方式...）
if directed:
    GAdj = sp.sparse.csr_array(GAdj.T)
G, node_map = dr.relabel_nodes(G, yield_map = True)
print(type(GAdj))
# 这里我们使用domirank算法找到最大特征值，使用黄金分割/二分法搜索空间，利用sigma > -1/lambN时的快速发散
# Here we find the maximum eigenvalue using the DomiRank algorithm and searching the space through a golden-ratio/bisection algorithm, taking advantage of the fast divergence when sigma > -1/lambN
# t1 = time.time()
lambN = dr.find_eigenvalue(GAdj, maxIter = 500, dt = 0.01, checkStep = 25) #sometimes you will need to change these parameters to get convergence有时候你需要改变这些参数来收敛
# t2 = time.time()
#重要提示：如果您不想比较domirank特征值计算与numpy计算的速度，请注释时间函数time的调用，但是对于特别大的网络请取消掉时间函数的调用
#IMPORTANT NOTE: for large graphs, comment out the lines below (23-26), along with lines (32-33).
#Please comment the part below (23-26) & (32-33) if you don't want a comparison with how fast the domirank eigenvalue computation is to the numpy computation.

print(f'\nThe found smallest eigenvalue was: lambda_N = {lambN}')
# print(f'\nOur single-threaded algorithm took: {t2-t1}s')

#注意，如果你只是执行dr.domirank(GAdj)并且没有传递最优的sigma，它将找到它自己。
#note, if you just perform dr.domirank(GAdj) and dont pass the optimal sigma, it will find it itself.
sigma, sigmaArray = dr.optimal_sigma(GAdj, analytical = analytical, endVal = lambN) #get the optimal sigma using the space (0, -1/lambN) as computed previously 使用之前计算的空间（0，-1/lambN）获取最佳sigma
print(f'\n The optimal sigma was found to be: {sigma*-lambN}/-lambda_N')


fig1 = plt.figure(1)
ourRange = np.linspace(0,1, sigmaArray.shape[0]) 
index = np.where(sigmaArray == sigmaArray.min())[0][-1]

plt.plot(ourRange, sigmaArray)
plt.plot(ourRange[index], sigmaArray[index], 'ro', mfc = 'none', markersize = 10)
plt.xlabel('sigma')
plt.ylabel('loss')


_, ourDomiRankDistribution = dr.domirank(GAdj, analytical = analytical, sigma = sigma) #generate the centrality using the optimal sigma
# ourDomiRankAttack = dr.generate_attack(ourDomiRankDistribution) #generate the attack using the centrality (descending)
# domiRankRobustness, domiRankLinks = dr.network_attack_sampled(GAdj, ourDomiRankAttack) #attack the network and get the largest connected component evolution

## 取消掉这里的注释：如果你想使用分析解，并且你的网络不是太大。
## UNCOMMENT HERE: to compute the analytical solution for the same sigma value (make sure your network is not too big.)

analyticalDomiRankDistribution = sp.sparse.linalg.spsolve(sigma*GAdj + sp.sparse.identity(GAdj.shape[0]), sigma*GAdj.sum(axis=-1)) #analytical solution to DR
analyticalDomiRankAttack = dr.generate_attack(analyticalDomiRankDistribution) #generate the attack using the centrality (descending)
domiRankRobustnessA, domiRankLinksA = dr.network_attack_sampled(GAdj, analyticalDomiRankAttack) #attack the network and get the largest connected component evolution

#generating the plot
fig2 = plt.figure(2)
ourRangeNew = np.linspace(0,1,domiRankRobustnessA.shape[0])
# plt.plot(ourRangeNew, domiRankRobustness, label = 'Recursive DR')
plt.plot(ourRangeNew, domiRankRobustnessA, label = 'Analytical DR') #UNCOMMENT HERE to plot the analyitcal solution
plt.legend()
plt.xlabel('fraction of nodes removed')
plt.ylabel('largest connected component')

# plt.show()
# in linux use png storage 
plt.savefig("resulta.png")