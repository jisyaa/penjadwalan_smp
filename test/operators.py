import random

def crossover(p1, p2):
    child = {}
    for k in p1:
        child[k] = p1[k] if random.random() < 0.5 else p2[k]
    return child

def mutate(ind, waktu_tersedia, rate=0.1):
    if random.random() > rate:
        return ind

    sid = random.choice(list(ind.keys()))
    ind[sid] = [random.choice(waktu_tersedia)["id_waktu"]]
    return ind
