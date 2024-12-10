########## 关联库 #############
import numpy as np
import scipy as sp
import scipy.sparse
import networkx as nx

########## 这里是一些分解网络和测试网络必需的函数 #############

def get_largest_component(G, strong = False):
    '''
    这里我们得到图的最大分支，无论是来自scipy.sparse还是来自networkX.Graph数据类型。
    改变参数`strong`来找到你想要的图的强连通分支或弱连通分支
    '''
    if type(G) == nx.classes.graph.Graph:
        if nx.is_directed(G) and strong == False:
            GMask = max(nx.weakly_connected_components(G), key = len)
        if nx.is_directed(G) and strong == True:
            GMask = max(nx.strongly_connected_components(G), key = len)
        else:
            GMask = max(nx.connected_components(G), key = len)
        G = G.subgraph(GMask)
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')
    return G

def relabel_nodes(G, yield_map = False):
    '''
    翻译：重新标记0，到len(G)节点
    如果你想保存哈希映射来检索节点id,`Yield_map`返回一个额外的字典类型输出。
    '''
    if yield_map == True:
        nodes = dict(zip(range(len(G)), G.nodes()))
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G, nodes
    else:
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G
    
def get_component_size(G, strong = False):
    '''
    here we get the largest component of a graph, either from scipy.sparse or from networkX.Graph datatype.
    1. The argument changes whether or not you want to find the strong or weak - connected components of the graph
    翻译：在这里，我们得到图的最大组成部分的大小（scipy.sparse或者networkX.Graph数据类型）。
    1.参数改变你想找到图的强连通分支还是弱连通分支
    '''
    if type(G) == nx.classes.graph.Graph: 
        if nx.is_directed(G) and strong == False:
            GMask = max(nx.weakly_connected_components(G), key = len)
        if nx.is_directed(G) and strong == True:
            GMask = max(nx.strongly_connected_components(G), key = len)
        else:
            GMask = max(nx.connected_components(G), key = len)
        G = G.subgraph(GMask)
        return len(GMask)        
    elif type(G) == scipy.sparse._arrays.csr_array:
        if strong == False:
            connection_type = 'weak'
        else:
            connection_type = 'strong'
        noComponent, lenComponent = sp.sparse.csgraph.connected_components(G, directed = True, connection = connection_type, return_labels = True)
        return np.bincount(lenComponent).max()
    else:
        raise TypeError('You must input a networkx.Graph Data-Type or scipy.sparse.csr array')
        
def get_link_size(G):
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        links = len(G.edges()) #convert to scipy sparse if it is a graph 
    elif type(G) == scipy.sparse._arrays.csr_array:
        links = G.sum()
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')
    return links

def remove_node(G, removedNode):
    '''
    翻译：从networkx.Graph类型中移除节点，或以数组形式零化边。
    '''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        if type(removedNode) == int:
            G.remove_node(removedNode)
        else:
            for node in removedNode:
                G.remove_node(node) #remove node in graph form
        return G
    elif type(G) == scipy.sparse._arrays.csr_array:
        diag = sp.sparse.csr_array(sp.sparse.eye(G.shape[0])) 
        diag[removedNode, removedNode] = 0 #set the rows and columns that are equal to zero in the sparse array
        G = diag @ G 
        return G @ diag
    
def generate_attack(centrality, node_map = False):
    '''
    翻译：根据中心性度量生成攻击———你可以输入node_map将攻击转换为正确的nodeID
    '''
    if node_map == False:
        node_map = range(len(centrality))
    else:
        node_map = list(node_map.values())
    zipped = dict(zip(node_map, centrality))
    attackStrategy = sorted(zipped, reverse = True, key = zipped.get)
    return attackStrategy

def network_attack_sampled(G, attackStrategy, sampling = 0):
    '''
    翻译：以采样的方式攻击网络...根据某种输入的attackStrategy，在每次节点移除后重新计算链接和最大组件，
    G：是输入图，最好是一个稀疏数组。
    注意：如果未设置采样，则默认为每1%采样一次，否则，采样是一个整数，等于每次采样时要跳过的节点数。
    例如，sampling = int(len(G)/100)将每1%采样一次
    '''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        GAdj = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph 
    else:
        GAdj = G.copy()
    
    if (sampling == 0) and (GAdj.shape[0] > 100):
        sampling = int(GAdj.shape[0]/100)
    if (sampling == 0) and (GAdj.shape[0] <= 100):
        sampling = 1
    N = GAdj.shape[0]
    initialComponent = get_component_size(GAdj)
    initialLinks = get_link_size(GAdj)
    m = GAdj.sum()/N
    componentEvolution = np.zeros(int(N/sampling))
    linksEvolution = np.zeros(int(N/sampling))
    j = 0 
    for i in range(N-1):
        if i % sampling == 0:
            if i == 0:
                componentEvolution[j] = get_component_size(GAdj)/initialComponent
                linksEvolution[j] = get_link_size(GAdj)/initialLinks
                j+=1 
            else:
                GAdj = remove_node(GAdj, attackStrategy[i-sampling:i])
                componentEvolution[j] = get_component_size(GAdj)/initialComponent
                linksEvolution[j] = get_link_size(GAdj)/initialLinks
                j+=1
    return componentEvolution, linksEvolution



######## domirank相关内容的开始 ####################

def domirank(G, analytical = True, sigma = -1, dt = 0.1, epsilon = 1e-5, maxIter = 1000, checkStep = 10):
    '''
    翻译：G是作为稀疏数组输入的图。
    这解决了论文中提出的动态方程：“DomiRank Centrality：通过节点优势揭示复杂网络的架构脆弱性”并产生以下输出：bool，DomiRankCentrality
    在这里，sigma需要事先选择。
    dt确定步长，通常，0.1对于大多数网络来说已经足够精细（可能会对度值极高的网络造成问题）
    maxIter是你在没有在之前收敛或发散之前搜索的深度。
    Checkstep是你在检查是否收敛或发散之前要走的步数。
    该算法与O(m)成比例，其中m是您的稀疏数组中的链接。
    '''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        G = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph 
    else:
        G = G.copy()
    if analytical == False:
        if sigma == -1:
            sigma, _ = optimal_sigma(G, analytical = False, dt=dt, epsilon=epsilon, maxIter = maxIter, checkStep = checkStep) 
        pGAdj = sigma*G.astype(np.float32)
        Psi = np.ones(pGAdj.shape[0]).astype(np.float32)/pGAdj.shape[0]
        maxVals = np.zeros(int(maxIter/checkStep)).astype(np.float32)
        dt = np.float32(dt)
        j = 0
        boundary = epsilon*pGAdj.shape[0]*dt
        for i in range(maxIter):
            tempVal = ((pGAdj @ (1-Psi)) - Psi)*dt
            Psi += tempVal.real
            if i% checkStep == 0:
                if np.abs(tempVal).sum() < boundary:
                    break
                maxVals[j] = tempVal.max()
                if i == 0:
                    initialChange = maxVals[j]
                if j > 0:
                    if maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2]:
                        return False, Psi
                j+=1

        return True, Psi
    else:
        if sigma == -1:
            sigma = optimal_sigma(G, analytical = True, dt=dt, epsilon=epsilon, maxIter = maxIter, checkStep = checkStep) 
        Psi = sp.sparse.linalg.spsolve(sigma*G + sp.sparse.identity(G.shape[0]), sigma*G.sum(axis=-1))
        return True, Psi
    
def find_eigenvalue(G, minVal = 0, maxVal = 1, maxDepth = 100, dt = 0.1, epsilon = 1e-5, maxIter = 100, checkStep = 10):
    '''
    翻译：G是作为稀疏数组输入的图。
    使用DomiRank算法找到邻接矩阵的最大负特征值。
    当前此函数仅是单线程的，因为二分算法只允许单线程执行。注意，这个算法略有不同，因为它利用了DomiRank在-1/lambN值更大的情况下发散的事实，因此它并不完全符合二分定理。我还没有测试过，以确定哪种确切值对于执行来说是最快的，但很快就会完成！
    一些说明：
    增加maxDepth以提高准确性。
    如果DomiRank在100次迭代内没有开始发散，则增加maxIter（以增加计算成本为代价，如果希望潜在提高准确性）。
    如果sigma的值太大，则减少checkstep以提高错误发现，但频繁减少该值时计算成本会变大（但计算成本可以忽略不计）。
    '''
    x = (minVal + maxVal)/G.sum(axis=-1).max()
    minValStored = 0
    for i in range(maxDepth):
        if maxVal - minVal < epsilon:
            break
        if domirank(G, False, x, dt, epsilon, maxIter, checkStep)[0]:
            minVal = x
            x = (minVal + maxVal)/2
            minValStored = minVal
        else:
            maxVal = (x + maxVal)/2
            x = (minVal + maxVal)/2
        if minVal == 0:
            print(f'Current Interval : [-inf, -{1/maxVal}]')
        else:
            print(f'Current Interval : [-{1/minVal}, -{1/maxVal}]')
    finalVal = (maxVal + minVal)/2
    return -1/finalVal



############## 本节用于寻找最优的sigma #######################

def process_iteration(q, i, analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling):
    tf, domiDist = domirank(spArray, analytical = analytical, sigma = sigma, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep)
    domiAttack = generate_attack(domiDist)
    ourTempAttack, __ = network_attack_sampled(spArray, domiAttack, sampling = sampling)
    finalErrors = ourTempAttack.sum()
    q.put((i, finalErrors))

def optimal_sigma(spArray, analytical = True, endVal = 0, startval = 0.000001, iterationNo = 100, dt = 0.1, epsilon = 1e-5, maxIter = 100, checkStep = 10, maxDepth = 100, sampling = 0):
    ''' 
    翻译：这部分通过搜索空间来找到最优的sigma，这里有一些新颖的参数：
    spArray：是网络的输入稀疏数组/矩阵。
    startVal：是你想要搜索的空间的起始值。
    endVal：是你想要搜索的空间的结束值（通常是特征值）
    iterationNo：你设置的lambN之间的空间划分的数量
    返回：函数返回sigma的值 - 分数（\sigma）/（-1*lambN）的分子
    '''
    if endVal == 0:
        endVal = find_eigenvalue(spArray, maxDepth = maxDepth, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep)
    import multiprocessing as mp
    endval = -0.9999/endVal
    tempRange = np.arange(startval, endval + (endval-startval)/iterationNo, (endval-startval)/iterationNo)
    processes = []
    q = mp.Queue()
    for i, sigma in enumerate(tempRange):
        p = mp.Process(target=process_iteration, args=(q, i, analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling))
        p.start()
        processes.append(p)

    results = [None] * len(tempRange)  # Initialize a results list

    #Join the processes and gather results from the queue翻译：加入进程并从队列中获取结果
    for p in processes:
        p.join()

    #Ensure that results are fetched from the queue after all processes are done翻译：确保在所有进程完成后从队列中获取结果
    while not q.empty():
        idx, result = q.get()
        results[idx] = result  # Store result in the correct order 以正确的顺序保存结果

    finalErrors = np.array(results)
    minEig = np.where(finalErrors == finalErrors.min())[0][-1]
    minEig = tempRange[minEig]
    return minEig, finalErrors

