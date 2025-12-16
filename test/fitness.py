from collections import defaultdict

def fitness(individual, sessions, waktu_map):
    penalty = 0
    guru_time = defaultdict(set)
    kelas_time = defaultdict(set)

    for s in sessions:
        slots = individual.get(s["sid"], [])
        days = set()

        for w in slots:
            if w in guru_time[s["id_guru"]]:
                penalty += 10_000
            if w in kelas_time[s["id_kelas"]]:
                penalty += 10_000

            guru_time[s["id_guru"]].add(w)
            kelas_time[s["id_kelas"]].add(w)
            days.add(waktu_map[w]["hari"])

        # session >1 jam HARUS kontigu
        if slots:
            if max(slots) - min(slots) + 1 != len(slots):
                penalty += 5_000

        # mapel >1 pertemuan harus beda hari
        if s["durasi"] > 1 and len(days) == 1:
            penalty += 2_000

    return 1_000_000 - penalty
