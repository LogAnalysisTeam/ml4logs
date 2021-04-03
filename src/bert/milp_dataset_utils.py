import mip as m
import numpy as np
from itertools import combinations
from collections import defaultdict
from typing import List

def compute_uniform_truncate_lenghts(lenghts: List[int], max_length:int) -> List[int]:
    if sum(lenghts) <= max_length:
        return lenghts
    else:
        model = m.Model()
        model.verbose = 0
        x = [model.add_var(var_type=m.INTEGER) for _ in lenghts]
        model += m.xsum(x) == max_length
        changed = defaultdict(lambda: model.add_var(var_type=m.BINARY))
        big_M = 2*max(lenghts)
        for i, val in enumerate(lenghts):
            x[i].lb = min(val, max_length//len(lenghts))
            x[i].ub = val
            if x[i].lb != x[i].ub:
                model += big_M*changed[i] >= val - x[i]
        d = {(i,j): model.add_var() for (i,j) in combinations((i for i in range(len(lenghts)) if x[i].lb != x[i].ub), r=2)}
        for i, j in d:
            model += d[i, j] >= x[i] - x[j]
            model += d[i, j] >= x[j] - x[i]

        model.objective = m.minimize(m.xsum(dif for dif in d.values()) + m.xsum(c for c in changed.values()))
        model.optimize()
        new_lengths = [int(val.x) for val in x]
        return new_lengths
