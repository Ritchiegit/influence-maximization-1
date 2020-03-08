''' Implements scalable algorithm for Linear Threshold (LT) model for directed graph G.
Each node in G is associated with a local DAG that is rooted at this node (LDAG(node)).
LDAG(node) 以node为结点的局部DAG有向无环图
Given a seed set S, this algorithm assumes that spread of influence from S to v
happens only within LDAG(v).
给定一个集合S，该算法假设影响力只在每个的LDAG上进行传播。
Then it adds nodes greedily. Each iteration it takes a node with the highest
incremental influence and update other nodes accordingly.
然后贪心地增加结点：每次迭代将带来嘴大influence增加值的结点加入已激活结点集合。
References:
[1] Chen et al. "Scalable Influence Maximization in Social Networks under Lienar Threshold Model"
核心：在局部图上进行传递的近似估计。

'''

from __future__ import division
from priorityQueue import PriorityQueue as PQ
import networkx as nx
from copy import deepcopy

# TODO write description for all functions and this script
def FIND_LDAG(G, v, t, Ew):
    '''
    Compute local DAG for vertex v.
    Reference: W. Chen "Scalable Influence Maximization in Social Networks under LT model" Algorithm 3
    INPUT:
        G -- networkx DiGraph object
        v -- vertex of G
        t -- parameter theta
        Ew -- influence weights of G
        NOTE: Since graph G can have multiple edges between u and v,
        total influence weight between u and v will be
        number of edges times influence weight of one edge.
    OUTPUT:
        D -- networkx DiGraph object that is also LDAG
    '''
    # intialize Influence of nodes
    Inf = PQ()
    Inf.add_task(v, -1)  # 存的是负值
    # print("Inf", Inf)
    # input()
    x, priority = Inf.pop_item()
    M = -priority  # 这里的M就是influence吧
    X = [x]

    D = nx.DiGraph()
    while M >= t:  # 如果影响力 大于 v的阈值还可以进行循环
        out_edges = G.out_edges([x], data=True)  # 原图 G

        for (v1,v2,edata) in out_edges:  # 如果
            if v2 in X:
                # D.add_edge(v1, v2, edata)
                D.add_edge(v1, v2, weight=edata["weight"])

        in_edges = G.in_edges([x])
        for (u,_) in in_edges:
            if u not in X:
                try:
                    [pr, _, _] = Inf.entry_finder[u]
                except KeyError:
                    pr = 0
                Inf.add_task(u, pr - G[u][x]['weight']*Ew[(u,x)]*M)  # M 是influence，G[u][x]['weight']和Ew[(u, x)] 要相乘？
        try:
            x, priority = Inf.pop_item()
        except KeyError:
            return D
        M = -priority
        X.append(x)

    return D


# def tsort(Dc, u, reach):
def tsort(Dc_input, u, reach):
    Dc = deepcopy(Dc_input)  # 避免对原来的有向图造成影响

    '''
    根据DAG进行拓扑排序
     Topological sort of DAG D with vertex u first.
     Note: procedure alters graph Dc (in the end no edges will be present)
     Input:
     Dc -- directed acyclic graph (nx.DiGraph)
      u -- root node (int)
    reach -- direction of topological sort (considered either incoming or outgoing edges) (string) "in" or "out"
    Output:
    L -- topological sort of nodes in Dc (list)
    '''
    assert (reach == "in" or reach == "out"), "reach argument can be either 'in' or 'out'."
    L = [u]
    print(L)
    if reach == "in":
        for node in L:
            # 从L中取一个 node 的入结点（上一级的结点） 存到L中，然后删去当前结点node
            # TODO test delete edge
            # in_nodes = map(lambda (v1, v2): v1, Dc.in_edges([node]))
            in_nodes = list(map(lambda edge: edge[0], Dc.in_edges([node])))
            # Dc.remove_edges_from(Dc.in_edges([node]))
            if node not in Dc.nodes:
                continue
            Dc.remove_node(node)  # 直接删除结点应该和删除起对应两条边效果一致吧 https://blog.csdn.net/qingqingpiaoguo/article/details/60570894
            for v in in_nodes:
                # if len(Dc.out_edges([v])) <= 1: # for self loops number of out_edges is 1, for other nodes is 0
                if v not in L and len(Dc.out_edges([v])) <= 1:  # 避免重复 加入L中
                    L.append(v)


    elif reach == "out": # the same just for outgoing edges
        for node in L:
            # TODO test delete edge
            # out_nodes = map(lambda (v1, v2): v2, Dc.out_edges([node]))
            out_nodes = list(map(lambda edge: edge[1], Dc.out_edges([node])))
            # Dc.remove_edges_from(Dc.out_edges([node]))
            if node not in Dc.nodes:
                continue
            Dc.remove_node(node)  # TODO 这一行代码发现了639
            for v in out_nodes:
                # if len(Dc.in_edges([v])) <= 1:
                if v not in L and len(Dc.in_edges([v])) <= 1:
                    L.append(v)
    if len(Dc.edges()):  # 如果把所有的点都删除，边应该也是空集吧
        raise(ValueError, "D has cycles. No topological order.")
    return L

def BFS_reach (D, u, reach):
    ''' Breadth-First search of nodes in D starting from node u.
    # 从有向无环图中提取出从u开始的子图
    Input:
    D -- directed acyclic graph (nx.DiGraph)
    u -- starting node (int)
    reach -- direction for which perform BFS
    Note:
        reach == "in" -- nodes that can reach u
        reach == "out" -- nodes that are reachable from u
    Output:
    Dc -- directed acyclic graph with edges in direction reach from u (nx.DiGraph)
    '''
    Dc = nx.DiGraph()
    if reach == "in":
        # 这样会给 Dc添加u 嗯 添加边的时候就会附带着添加节点呀
        Dc.add_edges_from(D.in_edges([u], data=True))  # u 的入边
        # TODO
        # 这个可以正常执行`
        # in_nodes = map(lambda (v1,v2): v1, D.in_edges([u]))
        in_nodes = list(map(lambda edge: edge[0], D.in_edges([u])))  # u的入边的结点
        # https://www.cnblogs.com/blackeyes1023/p/10954243.html
        # python 2 map 的返回值是一个列表，python 3 map的返回值是一个map对象
        # print(in_nodes)

        for node in in_nodes:
            Dc.add_edges_from(D.in_edges([node], data=True))  # 其实这里in_node 就是bfs里的队列
            # TODO
            # 将node的in_edge另一结点放在 队列 in_node 中
            # in_nodes.extend(filter(lambda v: v not in in_nodes, map(lambda (v1, v2): v1, D.in_edges([node]))))
            in_nodes.extend(list(filter(lambda v: v not in in_nodes, list(map(lambda edge: edge[0], D.in_edges([node]))))))
            # extend 是将可迭代元素中每一个元素都拿出来 https://thomas-cokelaer.info/blog/2011/03/post-2/
            # fliter python2.7 返回的是列表， python3返回迭代器对象 https://www.runoob.com/python/python-func-filter.html

    elif reach == "out": # the same just for outgoing edges
        Dc.add_edges_from(D.out_edges([u], data=True))
        # TODO
        # out_nodes = map(lambda (v1,v2): v2, D.out_edges([u]))
        out_nodes = list(map(lambda edge: edge[1], D.out_edges([u])))

        for node in out_nodes:
            Dc.add_edges_from(D.out_edges([node], data=True))
            # TODO
            # out_nodes.extend(filter(lambda v: v not in out_nodes, map(lambda (v1,v2): v2, D.out_edges([node]))))
            out_nodes.extend(list(filter(lambda v: v not in out_nodes, list(map(lambda edge: edge[1], D.out_edges([node]))))))

    return Dc

def computeAlpha(D, Ew, S, u, val=1):
    ''' Computing linear coefficients alphas between activation probabilities.
    Reference: [1] Algorithm 4
    Input:
    D -- directed acyclic graph (nx.DiGraph)
    Ew -- influence weights of edges (eg. uniform, random) (dict)
    S -- set of activated nodes (list)
    u -- starting node (int)
    val -- initialization value for u (int)
    Output:
    A -- linear coefficients alphas for all nodes in D (dict)
    '''
    A = dict()
    for node in D:
        A[(u,node)] = 0
    A[(u,u)] = val
    # compute nodes that can reach u in D
    # TODO print("u in D", u in D)
    Dc = BFS_reach(D, u, reach="in")  # 以u为源结点 的DAG子图  # 这里的写法不会将第一个结点u放进去
    # TODO print("u in Dc", u in Dc)  # 从 u 出发的结点里面居然没有u

    order = tsort(Dc, u, reach="in")  # 对LDAG(u) 进行拓扑排序  按照这一排序对 u结点影响的其他结点的 \alpha 进行更新
    # 原来的写法在BFS中可能会缺失 第一个结点u，并且因为 查找u相关的边进行删除 所以，不会察觉u的确实，
    # 但是这里 order[1:] 就不对了
    for node in order[1:]:  # miss first node that already has computed Alpha
        if node not in S + [u]:
            out_edges = D.out_edges([node], data=True)
            for (v1, v2, edata) in out_edges:
                assert v1 == node, 'First node should be the same'
                if v2 in order:
                    # print v1, v2, edata, Ew[(node, v2)], A[v2]
                    A[(u,node)] += edata['weight']*Ew[(node, v2)]*A[(u,v2)]
    return A

def computeActProb(D, Ew, S, u, val=1):
    # 计算激活概率
    ''' Computing activation probabilities for nodes in D.
    Reference: [1] Algorithm 2
    Input:
    D -- directed acyclic graph (nx.DiGraph)
    Ew -- influence weights of edges (eg. uniform, random) (dict)
    S -- set of activated nodes (list)
    u -- starting node (int)
    val -- initialization value for u (int)
    Output:
    ap -- activation probabilities of nodes in D (dict)
    '''
    ap = dict()
    for node in D:
        ap[(u,node)] = 0
    ap[(u,u)] = val
    Dc = BFS_reach(D, u, "out")
    order = tsort(Dc, u, "out")
    for node in order:
        if node not in S + [u]:
            in_edges = D.in_edges([node], data=True)
            for (v1, v2, edata) in in_edges:
                assert v2 == node, 'Second node should be the same'
                if v1 in order:
                    ap[(u,node)] += ap[(u,v1)]*Ew[(v1, node)]*edata['weight']
    return ap

def LDAG_heuristic(G, Ew, k, t):
    ''' LDAG algorithm for seed selection.
    Reference: [1] Algorithm 5
    Input:
    G -- directed graph (nx.DiGraph)
    Ew -- inlfuence weights of edges (eg. uniform, random) (dict)
    k -- size of seed set (int)
    t -- parameter theta for finding LDAG (0 <= t <= 1; typical value: 1/320) (int)
    Output:
    S -- seed set (list)
    '''
    # define variables
    S = []
    IncInf = PQ()
    for node in G:
        IncInf.add_task(node, 0)
    # IncInf = dict(zip(G.nodes(), [0]*len(G))) # in case of usage dict instead of PQ
    LDAGs = dict()  # u的local DAG
    InfSet = dict()  # 能够影响u的结点
    ap = dict()  # u的激活概率
    A = dict()  # \alpha

    print('Initialization phase')
    for v in G:
        LDAGs[v] = FIND_LDAG(G, v, t, Ew)
        # update influence set for each node in LDAGs[v] with its root
        for u in LDAGs[v]:
            InfSet.setdefault(u, []).append(v)  #
        alpha = computeAlpha(LDAGs[v], Ew, S, v)  # 计算alpha
        A.update(alpha)  # add new linear coefficients to A
        # update incremental influence of all nodes in LDAGs[v] with alphas
        for u in LDAGs[v]:
            ap[(v, u)] = 0 # additionally set initial activation probability (line 7)
            priority, _, _ = IncInf.entry_finder[u] # find previous value of IncInf
            IncInf.add_task(u, priority - A[(v, u)]) # and add alpha
            # IncInf[u] += A[(v, u)] # in case of using dict instead of PQ

    print('Main loop')
    for it in range(k):
        s, priority = IncInf.pop_item() # chose node with biggest incremental influence
        print(it+1, s, -priority)
        print("s", s)
        # print("InfSet", InfSet)
        for v in InfSet[s]: # for all nodes that s can influence
            print("InfSet[s]", InfSet[s])
            # s 能影响v，说明s在图LDAGv中
            if v not in S:
                D = LDAGs[v]
                # rint("D.in_edges", D.in_edges)
                # print(s in D)
                # update alpha_v_u for all u that can reach s in D (lines 17-22)
                alpha_v_s = A[(v,s)]
                # $\rho \backslash (S \cup \{ v\})$ 替换为$\rho \backslash (S \cup \{ s\})$；$\alpha_v()$替换为$\Delta \alpha_v()$
                dA = computeAlpha(D, Ew, S, s, val=-alpha_v_s)  # 报错 找不到结点 可能 还是只能删除边不能删除结点。
                # print("dA", dA)
                # input()
                for (s,u) in dA:
                    if u not in S + [s]: # don't update IncInf if it's already in S
                        A[(v,u)] += dA[(s,u)]
                        priority, _, _ = IncInf.entry_finder[u] # find previous value of incremental influence of u
                        IncInf.add_task(u, priority - dA[(s,u)]*(1 - ap[(v,u)])) # and update it accordingly
                # print("after for")
                # input()
                # update ap_v_u for all u reachable from s in D (lines 23-28)
                dap = computeActProb(D, Ew, S + [s], s, val=1-ap[(v,s)])  # Alg2 line 3~5
                for (s,u) in dap:
                    if u not in S + [s]:
                        ap[(v,u)] += dap[(s,u)]
                        priority, _, _ = IncInf.entry_finder[u] # find previous value of incremental influence of u
                        IncInf.add_task(u, priority + A[(v,u)]*dap[(s,u)]) # and update it accordingly
        S.append(s)
    return S
