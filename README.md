# LT and IC model with influence maximization algorithms in Python(Star 80+)

graphdata` 放置原始数据：边连接情况。

`graphs` 放置.gpickle文件。

`IC` 放置IC模型及其相关算法

​	`IC/ArbitraryP` 也是IC模型及其相关算法，

`LT` 放置LT模型及其相关算法



### 20200308

将LDAG算法从python 2 更新到 python 3

#### 问题记录

问题是这样开始的：



line 97 `tsort`函数中：`Dc.remove_edges_from(Dc.in_edges([node]))`在python 3 中报错：不能在循环中更改字典。

搜索无果后：

因为，tsort是对有向无环图进行拓扑排序，每次删除入度为0的结点。

所以我将 删除node的所有边，改成了删除node结点本身。（这样以node为结点的边也会被删除。） line 100 `Dc.remove_node(node)`



在运行 `LDAG_heuristic(G, Ew, k, t)`的主循环时报错

hep.txt数据集上 source = 639 时报错。

line 287 `dA = computeAlpha(D, Ew, S, s, val=-alpha_v_s)` 在计算$\Delta \alpha$时报错。原因是：

line 193 `order = tsort(Dc, u, reach="in")` 中 `Dc.remove_node(node)`找不到源结点`u`

很奇怪，`tsort` 里的排序的源结点u就是 `computeAlpha` 中的 `s`，Dc就是s能影响的结点v的DAG。

这个s既然能影响v，它就应在v的DAG中呀...



经过调试我发现在报错时，`FIND_LDAG`生成的图 D中包含结点s；但经过`BFS_reach`处理为Dc后，不仅不包含点s了，甚至整个Dc都为空...（在python2版本代码中，也会在相同的数据上出现一样的情况）

肯定是BFS_reach有情况。我本以为BFS_reach 只是走个过场，想直接略过这一步骤，结果如果不经过该函数，计算的影响力扩展度会从1300变成100，确实会影响结果。



究其根本是 `BFS_reach (D, u, reach):`函数的意义尚不明确

`FIND_LDAG(G, v, t, Ew)` 若已得到关于v点的Local DAG，为什么在使用前需要先用`BFS_reach`处理成`Dc`



#### 解决方法

缓兵之计：

当Dc中没有某一结点node时，不运行结点删除函数

line 98


```python
	if node not in Dc.nodes:
		continue
	Dc.remove_node(node)
```

marked 有空再看看