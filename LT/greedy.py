# -*- coding:UTF-8 -*-
'''
Implementation of greedy algorithm for LT model.
'''

__author__ = 'sergey'

from LT import avgLT
import time

def generalGreedy(G, Ew, k, iterations=20):
    start = time.time()
    S = []
    for i in range(k):
        Inf = dict() # influence for nodes not in S # 影响不在S中的结点
        for v in G:
            if v not in S:
                Inf[v] = avgLT(G, S + [v], Ew, iterations)  # TODO
        # u, val = max(Inf.iteritems(), key=lambda (k,val): val)  # 找到最大的值
        # u, ddv = max(dd.iteritems(), key=lambda (k,v): v)
        # u, ddv = max(iter(dd.items()), key=lambda k_v: k_v[1])
        u, val = max(iter(Inf.items()), key=lambda k_v: k_v[1])  # 对每个k_v取k_v的第一个元素
        print(i, u)
        S.append(u)
    return S