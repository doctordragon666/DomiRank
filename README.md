## DomiRank
复合中心性指标

### 项目简介

确定互连基础设施和复杂系统的关键要素对于确保系统功能和完整性至关重要。这项工作量化了网络节点在各自社区中的主导地位，引入了一个中心性指标 DomiRank，它通过可调参数整合了本地和全局拓扑信息。我们提出了一个用于 DomiRank 中心性的分析公式和一种高效的可并行化算法，使其适用于大规模网络。从网络的结构和功能的角度来看，DomiRank 值高的节点突出了脆弱的邻域，这些邻域的完整性和功能高度依赖于这些主导节点。为了强调支配性和脆弱性之间的这种关系，我们表明 DomiRank 在生成有针对性的攻击方面系统性地优于其他中心性指标，这些攻击有效地破坏了网络结构并破坏了其在合成和现实世界拓扑中的功能。此外，我们表明，基于 DomiRank 的攻击会在网络中造成更持久的损害，阻碍其反弹能力，从而损害系统弹性。DomiRank 中心性利用其定义中嵌入的竞争机制来揭示网络的脆弱性，为设计策略以减轻漏洞和增强关键基础设施的弹性铺平了道路。

### 项目使用效果图



### 安装说明

```shell
$pip install -r requirements.txt
```

### 使用说明

Change G to any network you want (networkx), or import any network and turn it into a scipy.sparse.csr_array() data structure. This will make sure the code runs flawlessly. 

Moreover, in the domirank.domirank() function, if you only pass the adjacency matrix (sparse) as an input, it will automatically compute the optimal sigma. However, you can also pass individual arguments, in order to create domiranks that will damage the network such that it is difficult to recover from, or, to simply, understand dynamics for high sigma (competition).

Finally, the network can be attacked according to any strategy, using the following function. domirank.network_attack_sampled(GAdj, attackStrategy), where GAdj is the adjacency matrix as a scipy.sparse.csr_array(), and the attack strategy is the ordering of the node removals (node-id). The node-id ordering can be generated from the centrality array by using the function domirank.generate_attack(centrality), where, centrality is an array of the centrality-distribution, ordered from (least to greatest in terms of node-id).

翻译：将 G 更改为您想要的任何网络（networkx），或者导入任何网络并将其转换为 scipy.sparse.csr_array() 数据结构。这将确保代码顺利运行。

此外，在 domirank.domirank() 函数中，如果您只传递邻接矩阵（稀疏）作为输入，它将自动计算最优 sigma。但是，您也可以传递单个参数，以创建会破坏网络并使其难以恢复的 domiranks，或者，仅为了简单地理解高 sigma（竞争）的动态。

最后，可以使用以下函数根据任何策略攻击网络。domirank.network_attack_sampled(GAdj, attackStrategy)，其中 GAdj 是作为 scipy.sparse.csr_array() 的邻接矩阵，攻击策略是节点删除的顺序（节点-id）。节点-id 排序可以通过使用函数 domirank.generate_attack(centrality) 从中心性数组生成，其中，中心性是一个按节点-id（从最小到最大）排序的中心性分布数组。


### 版权信息

该项目签署了MIT 授权许可，详情请参阅 LICENSE.md

### 鸣谢
本项目完全参考[DomiRank Centrality: revealing structural fragility of complex networks via node dominance](https://github.com/mengsig/DomiRank)实现，感谢原作者的分享！

如果您使用这个仓库，请引用以下手稿。
https://www.nature.com/articles/s41467-023-44257-0#citeas

Engsig, M., Tejedor, A., Moreno, Y. et al. DomiRank Centrality reveals structural fragility of complex networks via node dominance. Nat Commun 15, 56 (2024). https://doi.org/10.1038/s41467-023-44257-0
