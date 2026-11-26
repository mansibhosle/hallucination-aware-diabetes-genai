from collections import Counter
import math

def population_stability_index(reference: dict[str,float], current: dict[str,float], eps:float=1e-6)->float:
    keys=set(reference)|set(current); total=0.0
    for k in keys:
        r=max(reference.get(k,0),eps); c=max(current.get(k,0),eps)
        total += (c-r)*math.log(c/r)
    return total
