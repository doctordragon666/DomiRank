########## 关联库 #############
import numpy as np
import scipy as sp
import scipy.sparse
import networkx as nx


########## 这里是一些分解网络和测试网络必需的函数 #############


def get_largest_component(G, strong=False):
    '''
    获取图的最大连通片

    Parameters
    ----------
    `G` : networkx.Graph
        输入图。
    `strong` : bool, optional
        是否使用强连通分支。默认值为False。
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
    给图节点按照顺序添加数字标签(0,1,...,len(G)-1)

    Parameters
    ----------
    `G` : networkx.Graph
        输入图。
    `yield_map` : bool, optional
        是否返回节点映射。默认值为False。
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
    获取最大连通片的大小。

    Parameters
    ----------
    `G` : networkx.Graph or scipy.sparse.csr_array
        输入图或者稀疏矩阵。
    `strong` : bool, optional
        是否使用强连通分支。默认值为False。

    Returns
    -------
    size : int
        最大连通片的大小。
    '''
    if type(G) == nx.classes.graph.Graph:
        return len(get_largest_component(G))
    elif type(G) == scipy.sparse.csr_array:
        if strong == False:
            connection_type = 'weak'
        else:
            connection_type = 'strong'
        nComponent, lenComponent = sp.sparse.csgraph.connected_components(
            G, directed=True, connection=connection_type, return_labels=True)
        return np.bincount(lenComponent).max()
    else:
        raise TypeError(
            'You must input a networkx.Graph Data-Type or scipy.sparse.csr array')


def get_link_size(G):
    """
    获取图中的边数量。

    Parameters
    ----------
    `G` : networkx.Graph or scipy.sparse.csr_array
        输入图。

    Returns
    -------
    links : int
        图中的边数量。
    """
    if type(G) == nx.classes.graph.Graph:
        links = len(G.edges())
    elif type(G) == scipy.sparse.csr_array:
        links = G.sum()
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')
    return links


def remove_node(G, removedNode):
    '''
    移除节点

    Parameters
    ----------
    `G` : networkx.Graph or scipy.sparse.csr_array
        输入图。
    `removedNode` : int or array
        要移除的节点的标签或者序号。

    Returns
    -------
    G : networkx.Graph or scipy.sparse.csr_array
        移除节点后的图。
    '''
    if type(G) == nx.classes.graph.Graph:
        if type(removedNode) == int:
            G.remove_node(removedNode)
        else:
            for node in removedNode:
                G.remove_node(node)
        return G
    elif type(G) == scipy.sparse.csr_array:
        # 线性代数，乘以对角矩阵来去除节点
        diag = sp.sparse.csr_array(sp.sparse.eye(G.shape[0]))
        diag[removedNode, removedNode] = 0
        G = diag @ G
        return G @ diag


def generate_attack(centrality, node_map=False):
    '''
    根据中心性度量生成攻击策略（指定要移除的节点）

    Parameters
    ----------
    `centrality` : array
        中心性度量。
    `node_map` : dict, optional
        节点映射。默认值为False。

    Returns
    -------
    attackStrategy : array
        攻击策略。
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
    以采样的方式攻击网络，在每次节点移除后重新计算链接和最大组件，返回组件和链接的变化。
    注意：采样是一个整数，等于每次采样时要跳过的节点数。如果未设置采样，则默认为每1%采样一次

    Parameters
    ----------
    `G` : networkx.Graph or scipy.sparse.csr_array
        输入图，最好是稀疏矩阵。
    `attackStrategy` : array
        攻击策略，由`generate_attack`生成。
    `sampling` : int, optional
        采样率。默认值为0时，会进行计算，如果G的节点数大于100，则采样率为节点数除以100，否则为1。

    Returns
    -------
    componentEvolution : array
        组件变化数组。
    linksEvolution : array
        链接变化数组。
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

    # 获取GAdj的初始组件大小,初始化组件变化数组
    initialComponent, initialLinks = get_component_size(
        GAdj), get_link_size(GAdj)
    componentEvolution, linksEvolution = np.zeros(
        int(N/sampling)), np.zeros(int(N/sampling))
    componentEvolution[0], linksEvolution[0] = 1, 1

    for j in range(1, int(N/sampling)):
        if sampling*j <= N:
            GAdj = remove_node(GAdj, attackStrategy[sampling*(j-1):sampling*j])
        else:
            GAdj = remove_node(GAdj, attackStrategy[sampling*(j-1):])

        componentEvolution[j] = get_component_size(
            GAdj)/initialComponent
        linksEvolution[j] = get_link_size(GAdj)/initialLinks

    return componentEvolution, linksEvolution


######## domirank相关内容的开始 ####################

def domirank_by_annalytical(G, sigma=-1, dt=0.1, epsilon=1e-5, maxIter=1000, checkStep=10):
    '''
    使用数学的解析方法计算论文里面提出的方程解：Gamma = theta sigma(sigma A + I)^{-1} A
    该算法与O(m)成比例，其中m是您的稀疏数组中的链接。

    Parameters
    ----------
    `G` : networkx.Graph or scipy.sparse.csr_array
        输入图，最好是稀疏矩阵。
    `sigma` : float, optional
        sigma值。默认值为-1，表示使用optimal_sigma函数计算最优sigma值。
    `dt` : float, optional
        步长。默认值为0.1。0.1对于大多数网络来说已经足够精细（可能会对度值极高的网络造成问题）
    `epsilon` : float, optional
        收敛阈值。默认值为1e-5。
    `maxIter` : int, optional
        在没有确定收敛或发散之前最大迭代次数。默认值为1000。
    `checkStep` : int, optional
        检查是否收敛或发散要走的步数。默认值为10。

    Returns
    -------
    Psi : array
        每个节点的Domirank数组。
    '''
    if type(G) == nx.classes.graph.Graph:
        G = nx.to_scipy_sparse_array(G)
    if sigma == -1:
        sigma = optimal_sigma(
            G, analytical=True, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)
    # 使用稀疏矩阵求解器求解sigma*G + sp.sparse.identity(G.shape[0])的逆矩阵，并乘以sigma*G.sum(axis=-1)
    Psi = sp.sparse.linalg.spsolve(
        sigma*G + sp.sparse.identity(G.shape[0]), sigma*G.sum(axis=-1))
    return Psi


def domirank_by_recursive(G, sigma=-1, dt=0.1, epsilon=1e-5, maxIter=1000, checkStep=10):
    '''
    使用迭代的方法计算论文里面提出的方程解：Gamma(t+dt) = Gamma(t) + beta[sigma A(E-Gamma(t))-Gamma(t)]dt
    该算法与O(m)成比例，其中m是您的稀疏数组中的链接。

    Parameters
    ----------
    `G` : networkx.Graph or scipy.sparse.csr_array
        输入图，最好是稀疏矩阵。
    `sigma` : float, optional
        sigma值。默认值为-1，表示使用optimal_sigma_thread函数计算最优sigma值。
    `dt` : float, optional
        步长。默认值为0.1。0.1对于大多数网络来说已经足够精细（可能会对度值极高的网络造成问题）
    `epsilon` : float, optional
        收敛阈值。默认值为1e-5。
    `maxIter` : int, optional
        在没有确定收敛或发散之前最大迭代次数。默认值为1000。
    `checkStep` : int, optional
        检查是否收敛或发散要走的步数。默认值为10。

    Returns
    -------
    converged : bool
        是否收敛。
    Psi : array
        每个节点的Domirank数组。
    '''
    if type(G) == nx.classes.graph.Graph:
        G = nx.to_scipy_sparse_array(G)

    # 如果sigma为-1，则调用optimal_sigma函数计算最优sigma值
    if sigma == -1:
        sigma, _ = optimal_sigma(
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
            if j > 1:
                if maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2]:
                    return False, Psi
            j += 1

    return True, Psi

############## 本节用于寻找最优的sigma #######################


def find_eigenvalue(G, **args):
    '''
    计算G的最大特征值，因为原作者使用domirank算法计算的，这里为了简便，转换为scipy的稀疏矩阵来计算特征值。
    同时为了保持与原作者的API一致，没有删掉这个函数。

    Parameters
    ----------
    `G` : networkx.Graph or scipy.sparse.csr_array
        输入图，最好是稀疏矩阵。
    `**args` : dict
        其他参数。

    Returns
    -------
    eigenvalue : float
        最大特征值。
    '''
    if type(G) == nx.classes.graph.Graph:
        G = nx.to_scipy_sparse_array(G)
    G = G.astype(np.float64)
    # 计算G的特征值
    eigenvalues, _ = sp.sparse.linalg.eigsh(G, k=5, which='LA')
    return -1/max(eigenvalues)  # 返回最终值

def find_eigenvalue(G, minVal = 0, maxVal = 1, maxDepth = 100, dt = 0.1, epsilon = 1e-5, maxIter = 100, checkStep = 10):
    '''
    G: is the input graph as a sparse array.
    Finds the largest negative eigenvalue of an adjacency matrix using the DomiRank algorithm.
    Currently this function is only single-threaded, as the bisection algorithm only allows for single-threaded
    exection. Note, that this algorithm is slightly different, as it uses the fact that DomiRank diverges
    at values larger than -1/lambN to its benefit, and thus, it is not exactly bisection theorem. I haven't
    tested in order to see which exact value is the fastest for execution, but that will be done soon!
    Some notes:
    Increase maxDepth for increased accuracy.
    Increase maxIter if DomiRank doesn't start diverging within 100 iterations -- i.e. increase at the expense of 
    increased computational cost if you want potential increased accuracy.
    Decrease checkstep for increased error-finding for the values of sigma that are too large, but higher compcost
    if you are frequently less than the value (but negligible compcost).
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

def process_iteration_thread(x):
    '''
    处理每个线程的迭代函数，作用是根据传入的参数计算domiRank，返回曲线下的面积，越小说明攻击的效果越好。

    Parameters
    ----------
    `x` : tuple
        传入的参数，包括当前迭代的索引、sigma值、稀疏矩阵、最大迭代次数、检查步长、dt值、epsilon值和采样率。

    Returns
    -------
    `tuple`
        返回一个元组，包括当前迭代的索引和最终误差。
    '''
    i, analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling = x
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
    return (i, finalErrors)


def optimal_sigma(spArray, analytical=True, endVal=0, startval=0.000001, iterationNo=100, dt=0.1, epsilon=1e-5, maxIter=100, checkStep=10, sampling=0):
    ''' 
    搜索空间来找到最优的sigma
    spArray：是网络的输入稀疏数组/矩阵。
    startVal：是你想要搜索的空间的起始值。
    endVal：是你想要搜索的空间的结束值（通常是特征值）
    iterationNo：你设置的lambN之间的空间划分的数量

    Parameters
    ----------
    `spArray` : scipy.sparse.csr_array
        稀疏矩阵。
    `analytical` : bool, optional
        是否使用解析方法。默认值为True。
    `endVal` : float, optional
        结束值。默认值为0。
    `startval` : float, optional
        起始值。默认值为0.000001。
    `iterationNo` : int, optional
        迭代次数。默认值为100。
    `dt` : float, optional
        步长。默认值为0.1。
    `epsilon` : float, optional
        收敛阈值。默认值为1e-5。
    `maxIter` : int, optional
        最大迭代次数。默认值为100。
    `checkStep` : int, optional
        检查步长。默认值为10。
    `sampling` : int, optional
        采样率。默认值为0。
    '''
    # 如果endVal为0，则调用find_eigenvalue函数计算endVal
    if endVal == 0:
        endVal = find_eigenvalue(spArray)
    # 计算endval的值
    endval = -0.9999/endVal  # 这个是作者为了转换为浮点数使用的
    # 计算tempRange的值
    originRange = np.arange(startval, endval + (endval-startval) /
                          iterationNo, (endval-startval)/iterationNo)

    import multiprocessing as mp
    tempRange = [(i, analytical, sigma, spArray, maxIter, checkStep,
                  dt, epsilon, sampling) for i, sigma in enumerate(originRange)]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_iteration_thread, tempRange)

    finalErrors = np.array([result[1] for result in results])
    minEig = originRange[np.where(finalErrors == finalErrors.min())[0][-1]]
    return minEig, finalErrors
