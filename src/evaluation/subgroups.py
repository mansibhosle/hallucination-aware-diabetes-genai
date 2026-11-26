from collections import defaultdict

def by_category(rows:list[dict], key="category"):
    groups=defaultdict(list)
    for r in rows: groups[r.get(key,"unknown")].append(r)
    out={}
    for k,items in groups.items():
        out[k]={"n":len(items),"accuracy":sum(bool(x.get("correct")) for x in items)/max(len(items),1),"abstain_rate":sum(str(x.get("decision","")).startswith("ABSTAIN") for x in items)/max(len(items),1)}
    return out
