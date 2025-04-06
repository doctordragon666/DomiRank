########## 关联库 #############
import asyncio
import numpy as np
import scipy as sp
import scipy.sparse
import networkx as nx
import multiprocessing as mp # optimal_sigmaw函数需要

########## 这里是一些分解网络和测试网络必需的函数 #############


def get_largest_component(G, strong=False):
    '''
    这里我们得到图的最大分支，无论是来自scipy.sparse还是来自networkX.Graph数据类型。
    改变参数`strong`来找到你想要的图的强连通分支或弱连通分支
    '''
    if type(G) == nx.classes.graph.Graph:
        if nx.is_directed(G) and strong == False:
            GMask = max(nx.weakly_connected_components(G), key=len)
        if nx.is_directed(G) and strong == True:
            GMask = max(nx.strongly_connected_components(G), key=len)
        else:
            GMask = max(nx.connected_components(G), key=len)
        G = G.subgraph(GMask)
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')
    return G


def relabel_nodes(G, yield_map=False):
    '''
    重新标记0，到len(G)节点
    如果你想保存哈希映射来检索节点id,`Yield_map`返回一个额外的字典类型输出。
    '''
    if yield_map:
        nodes = dict(zip(range(len(G)), G.nodes()))
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G, nodes
    else:
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G


def get_component_size(G, strong=False):
    '''
    在这里，我们得到图的最大组成部分的大小（scipy.sparse或者networkX.Graph数据类型）。
    1.参数改变你想找到图的强连通分支还是弱连通分支
    '''
    if type(G) == nx.classes.graph.Graph:
        return len(get_largest_component(G))
    elif type(G) == scipy.sparse._arrays.csr_array:
        if strong == False:
            connection_type = 'weak'
        else:
            connection_type = 'strong'
        noComponent, lenComponent = sp.sparse.csgraph.connected_components(
            G, directed=True, connection=connection_type, return_labels=True)
        return np.bincount(lenComponent).max()
    else:
        raise TypeError(
            'You must input a networkx.Graph Data-Type or scipy.sparse.csr array')


def get_link_size(G):
    """
    获取图中的边数。
    """
    if type(G) == nx.classes.graph.Graph:
        links = len(G.edges())
    elif type(G) == scipy.sparse._arrays.csr_array:
        links = G.sum()
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')
    return links


def remove_node(G, removedNode):
    '''
    移除节点，或以数组形式零化边。
    '''
    if type(G) == nx.classes.graph.Graph:  # check if it is a networkx Graph
        if type(removedNode) == int:
            G.remove_node(removedNode)
        else:
            for node in removedNode:
                G.remove_node(node)  # remove node in graph form
        return G
    elif type(G) == scipy.sparse._arrays.csr_array:
        diag = sp.sparse.csr_array(sp.sparse.eye(G.shape[0]))
        diag[removedNode, removedNode] = 0
        G = diag @ G
        return G @ diag


def generate_attack(centrality, node_map=False):
    '''
    翻译：根据中心性度量生成攻击———输入node_map将攻击转换为实际的nodeID
    '''
    if node_map == False:
        node_map = range(len(centrality))
    else:
        node_map = list(node_map.values())
    zipped = dict(zip(node_map, centrality))
    attackStrategy = sorted(zipped, reverse=True, key=zipped.get)
    return attackStrategy


def network_attack_sampled(G, attackStrategy, sampling=0):
    '''
    翻译：以采样的方式攻击网络...根据某种输入的attackStrategy，在每次节点移除后重新计算链接和最大组件，
    G：是输入图，最好是一个稀疏数组。
    注意：如果未设置采样，则默认为每1%采样一次，否则，采样是一个整数，等于每次采样时要跳过的节点数。
    例如，sampling = int(len(G)/100)将每1%采样一次
    '''
    # 如果G是图类型，则将G转换为稀疏矩阵
    if type(G) == nx.classes.graph.Graph:
        GAdj = nx.to_scipy_sparse_array(G)
    else:
        GAdj = G.copy()

    # 如果采样率为0且GAdj的节点数大于100，则将采样率设置为节点数除以100
    if (sampling == 0) and (GAdj.shape[0] > 100):
        sampling = int(GAdj.shape[0]/100)
    # 如果采样率为0且GAdj的节点数小于等于100，则将采样率设置为1
    if (sampling == 0) and (GAdj.shape[0] <= 100):
        sampling = 1
    # 获取GAdj的节点数
    N = GAdj.shape[0]
    # 获取GAdj的初始组件大小
    initialComponent, initialLinks = get_component_size(
        GAdj), get_link_size(GAdj)
    # 计算GAdj的平均链接数
    m = GAdj.sum()/N
    # 初始化组件变化数组
    componentEvolution = np.zeros(int(N/sampling))
    # 初始化链接变化数组
    linksEvolution = np.zeros(int(N/sampling))
    # 初始化计数器
    j = 0
    # 遍历GAdj的节点
    for i in range(N-1):
        # 如果计数器与采样率相等，则进行操作
        if i % sampling == 0:
            # 如果计数器为0，则计算初始组件和链接大小
            if i == 0:
                componentEvolution[j] = get_component_size(
                    GAdj)/initialComponent
                linksEvolution[j] = get_link_size(GAdj)/initialLinks
                j += 1
            else:
                # 否则，移除节点并计算组件和链接大小
                GAdj = remove_node(GAdj, attackStrategy[i-sampling:i])
                componentEvolution[j] = get_component_size(
                    GAdj)/initialComponent
                linksEvolution[j] = get_link_size(GAdj)/initialLinks
                j += 1
    # 返回组件和链接变化数组
    return componentEvolution, linksEvolution


######## domirank相关内容的开始 ####################

def domirank_by_annalytical(G, sigma=-1, dt=0.1, epsilon=1e-5, maxIter=1000, checkStep=10):
    '''
    G是作为稀疏数组输入的图。
    这解决了论文中提出的动态方程：“DomiRank Centrality：通过节点优势揭示复杂网络的架构脆弱性”并产生以下输出：DomiRankCentrality
    在这里，sigma需要事先选择。
    dt确定步长，通常，0.1对于大多数网络来说已经足够精细（可能会对度值极高的网络造成问题）
    maxIter是你在没有在之前收敛或发散之前搜索的深度。
    Checkstep是你在检查是否收敛或发散之前要走的步数。
    该算法与O(m)成比例，其中m是您的稀疏数组中的链接。
    '''
    if type(G) == nx.classes.graph.Graph:
        G = nx.to_scipy_sparse_array(G)
    # 如果sigma为-1，则调用optimal_sigma函数计算最优sigma值
    if sigma == -1:
        sigma = optimal_sigman(
            G, analytical=True, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)
    # 使用稀疏矩阵求解器求解sigma*G + sp.sparse.identity(G.shape[0])的逆矩阵，并乘以sigma*G.sum(axis=-1)
    Psi = sp.sparse.linalg.spsolve(
        sigma*G + sp.sparse.identity(G.shape[0]), sigma*G.sum(axis=-1))
    return Psi


def domirank_by_recursive(G, sigma=-1, dt=0.1, epsilon=1e-5, maxIter=1000, checkStep=10):
    '''
    G是作为稀疏数组输入的图。
    这解决了论文中提出的动态方程：“DomiRank Centrality：通过节点优势揭示复杂网络的架构脆弱性”并产生以下输出：bool，DomiRankCentrality
    在这里，sigma需要事先选择。
    dt确定步长，通常，0.1对于大多数网络来说已经足够精细（可能会对度值极高的网络造成问题）
    maxIter是你在没有在之前收敛或发散之前搜索的深度。
    Checkstep是你在检查是否收敛或发散之前要走的步数。
    该算法与O(m)成比例，其中m是您的稀疏数组中的链接。
    '''
    if type(G) == nx.classes.graph.Graph:
        G = nx.to_scipy_sparse_array(G)
    else:
        G = G.copy()

    # 如果sigma为-1，则调用optimal_sigma函数计算最优sigma值
    if sigma == -1:
        sigma, _ = optimal_sigman(
            G, analytical=False, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)
    pGAdj = sigma*G.astype(np.float32)
    # 初始化Psi为1/n，n为pGAdj的行数
    Psi = np.ones(pGAdj.shape[0]).astype(np.float32)/pGAdj.shape[0]
    # 初始化maxVals为0，长度为maxIter/checkStep
    maxVals = np.zeros(int(maxIter/checkStep)).astype(np.float32)
    dt = np.float32(dt)
    j = 0
    boundary = epsilon*pGAdj.shape[0]*dt
    for i in range(maxIter):
        tempVal = ((pGAdj @ (1-Psi)) - Psi)*dt
        Psi += tempVal
        if i % checkStep == 0:
            if np.abs(tempVal).sum() < boundary:
                break
            maxVals[j] = tempVal.max()
            # if j > 0: 这里犯了一个错误，这里不一定前者不成立，
            if j > 1:
                if maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2]:
                    return False, Psi
            j += 1

    return True, Psi

############## 本节用于寻找最优的sigma #######################


def find_eigenvaluen(G, minVal=0, maxVal=1, maxDepth=100, dt=0.1, epsilon=1e-5, maxIter=100, checkStep=10):
    '''
    翻译：G是作为稀疏数组输入的图。
    使用DomiRank算法找到邻接矩阵的最大负特征值。
    当前此函数仅是单线程的，因为二分算法只允许单线程执行。注意，这个算法略有不同，因为它利用了DomiRank在-1/lambN值更大的情况下发散的事实，因此它并不完全符合二分定理。我还没有测试过，以确定哪种确切值对于执行来说是最快的，但很快就会完成！
    一些说明：
    增加maxDepth以提高准确性。
    如果DomiRank在100次迭代内没有开始发散，则增加maxIter（以增加计算成本为代价，如果希望潜在提高准确性）。
    如果sigma的值太大，则减少checkstep以提高错误发现，但频繁减少该值时计算成本会变大（但计算成本可以忽略不计）。
    '''
    x = (minVal + maxVal)/G.sum(axis=-1).max()  # 计算初始值x
    # minValStored = 0  # 初始化最小值存储变量
    for i in range(maxDepth):  # 循环maxDepth次
        if maxVal - minVal < epsilon:  # 如果最大值和最小值的差小于epsilon，则跳出循环
            break
        # 如果domirank函数返回True
        if domirank_by_recursive(G, x, dt, epsilon, maxIter, checkStep)[0]:
            minVal = x  # 更新最小值
            x = (minVal + maxVal)/2  # 更新x
            # minValStored = minVal  # 更新最小值存储变量
        else:
            maxVal = (x + maxVal)/2  # 更新最大值
            x = (minVal + maxVal)/2  # 更新x
        if minVal == 0:  # 如果最小值为0
            print(f'Current Interval : [-inf, -{1/maxVal}]')  # 打印当前区间
        else:
            print(f'Current Interval : [-{1/minVal}, -{1/maxVal}]')  # 打印当前区间
    finalVal = (maxVal + minVal)/2  # 计算最终值
    return -1/finalVal  # 返回最终值


def process_iterationn(q, i, analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling):
    # 计算domiRank和domiDist
    if analytical:
        domiDist = domirank_by_annalytical(spArray, sigma=sigma,
                                           dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)
    else:
        _, domiDist = domirank_by_recursive(spArray, sigma=sigma,
                                            dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)
    # 生成攻击
    domiAttack = generate_attack(domiDist)
    # 在采样网络上进行攻击
    ourTempAttack, __ = network_attack_sampled(
        spArray, domiAttack, sampling=sampling)
    # 计算最终误差
    finalErrors = ourTempAttack.sum()
    # 将结果放入队列
    q.put((i, finalErrors))


def optimal_sigman(spArray, analytical=True, endVal=0, startval=0.000001, iterationNo=100, dt=0.1, epsilon=1e-5, maxIter=100, checkStep=10, maxDepth=100, sampling=0):
    ''' 
    翻译：搜索空间来找到最优的sigma
    spArray：是网络的输入稀疏数组/矩阵。
    startVal：是你想要搜索的空间的起始值。
    endVal：是你想要搜索的空间的结束值（通常是特征值）
    iterationNo：你设置的lambN之间的空间划分的数量
    返回：函数返回sigma的值 - 分数（\sigma）/（-1*lambN）的分子
    '''
    # 如果endVal为0，则调用find_eigenvalue函数计算endVal
    if endVal == 0:
        endVal = find_eigenvaluen(spArray, maxDepth=maxDepth, dt=dt,
                                 epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)
    # 计算endval的值
    endval = -0.9999/endVal
    # 计算tempRange的值
    tempRange = np.arange(startval, endval + (endval-startval) /
                          iterationNo, (endval-startval)/iterationNo)
    # 创建一个进程列表
    processes = []
    # 创建一个队列
    q = mp.Queue()
    # 遍历tempRange，创建进程
    for i, sigma in enumerate(tempRange):
        p = mp.Process(target=process_iterationn, args=(
            q, i, analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling))
        p.start()
        processes.append(p)

    # 创建一个结果列表
    results = [None] * len(tempRange)

    # 等待所有进程结束
    for p in processes:
        p.join()
    # 从队列中获取结果
    while not q.empty():
        idx, result = q.get()
        results[idx] = result

    # 将结果转换为numpy数组
    finalErrors = np.array(results)
    # 找到最小误差的索引
    minEig = np.where(finalErrors == finalErrors.min())[0][-1]
    # 找到最小误差对应的tempRange的值
    minEig = tempRange[minEig]
    # 返回最小误差对应的tempRange的值和所有误差
    return minEig, finalErrors

# def optimal_sigmaw(spArray, analytical=True, endVal=0, startval=0.000001, iterationNo=100, dt=0.1, epsilon=1e-5, maxIter=100, checkStep=10, maxDepth=100, sampling=0):
#     ''' 
#     翻译：搜索空间来找到最优的sigma
#     spArray：是网络的输入稀疏数组/矩阵。
#     startVal：是你想要搜索的空间的起始值。
#     endVal：是你想要搜索的空间的结束值（通常是特征值）
#     iterationNo：你设置的lambN之间的空间划分的数量
#     返回：函数返回sigma的值 - 分数（\sigma）/（-1*lambN）的分子
#     '''
#     # 如果endVal为0，则调用find_eigenvalue函数计算endVal
#     if endVal == 0:
#         endVal = find_eigenvaluen(spArray, maxDepth=maxDepth, dt=dt,
#                                  epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)
#     # 导入multiprocessing模块
#     import multiprocessing as mp
#     # 计算endval的值
#     endval = -0.9999/endVal
#     # 计算tempRange的值
#     tempRange = np.arange(startval, endval + (endval-startval) /
#                           iterationNo, (endval-startval)/iterationNo)
#     # 创建一个进程列表
#     processes = []
#     # 创建一个队列
#     q = mp.Queue()
#     # 遍历tempRange，创建进程
#     for i, sigma in enumerate(tempRange):
#         p = mp.Process(target=process_iterationn, args=(
#             q, i, analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling))
#         p.start()
#         processes.append(p)

#     # 创建一个结果列表
#     results = [None] * len(tempRange)

#     # 等待所有进程结束
#     for p in processes:
#         p.join()
#     # 从队列中获取结果
#     while not q.empty():
#         idx, result = q.get()
#         results[idx] = result

#     # 将结果转换为numpy数组
#     finalErrors = np.array(results)
#     # 找到最小误差的索引
#     minEig = np.where(finalErrors == finalErrors.min())[0][-1]
#     # 找到最小误差对应的tempRange的值
#     minEig = tempRange[minEig]
#     # 返回最小误差对应的tempRange的值和所有误差
#     return minEig, finalErrors