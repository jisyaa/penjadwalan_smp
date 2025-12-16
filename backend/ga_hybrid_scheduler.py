# ============================================================
# GA HYBRID SCHOOL TIMETABLING
# Constructive Heuristic + Genetic Algorithm
# ============================================================

import random
import math
import os
import csv
from collections import defaultdict
from datetime import datetime

from sqlalchemy import create_engine, text

# =========================
# CONFIG
# =========================
DB_CONFIG = {
    "user": "root",
    "password": "",
    "host": "localhost",
    "db": "db_penjadwalan"
}

POP_SIZE = 30
GENERATIONS = 150
MUTATION_RATE = 0.15
ELITISM = 2

DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat"]

# =========================
# DB
# =========================
def get_engine():
    url = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['db']}"
    return create_engine(url, future=True)

def read_tables():
    engine = get_engine()
    tables = {}
    with engine.connect() as conn:
        for t in ["kelas", "mapel", "guru", "guru_mapel", "waktu"]:
            tables[t] = conn.execute(text(f"SELECT * FROM {t}")).mappings().all()
    return tables

# =========================
# SESSION SPLITTER
# =========================
def split_sessions(jam):
    """
    ATURAN KAMU:
    1 -> [1]
    3 -> [3]
    4 -> [2,2]
    5 -> [3,2]
    6 -> [3,3]
    """
    if jam == 1:
        return [1]
    if jam == 3:
        return [3]
    if jam == 4:
        return [2,2]
    if jam == 5:
        return [3,2]
    if jam == 6:
        return [3,3]
    raise ValueError(f"Jam tidak didukung: {jam}")

# =========================
# BUILD SESSIONS
# =========================
def build_sessions(tables):
    sessions = []
    sid = 0

    guru_mapel = defaultdict(list)
    for gm in tables["guru_mapel"]:
        guru_mapel[(gm["id_kelas"], gm["id_mapel"])].append(gm["id_guru"])

    for k in tables["kelas"]:
        for m in tables["mapel"]:
            key = (k["id_kelas"], m["id_mapel"])
            if key not in guru_mapel:
                continue

            parts = split_sessions(m["jam_per_minggu"])
            guru_id = guru_mapel[key][0]

            for p in parts:
                sessions.append({
                    "sid": sid,
                    "kelas": k["nama_kelas"],
                    "id_kelas": k["id_kelas"],
                    "mapel": m["nama_mapel"],
                    "id_mapel": m["id_mapel"],
                    "guru": guru_id,
                    "length": p
                })
                sid += 1
    return sessions

# =========================
# SLOT TEMPLATE
# =========================
def build_slots(tables):
    slots = []
    for w in tables["waktu"]:
        if w["keterangan"] is None or w["keterangan"] == "":
            slots.append((w["hari"], int(w["jam_ke"])))
    slots.sort(key=lambda x: (DAYS.index(x[0]), x[1]))
    return slots

# =========================
# CONSTRUCTIVE INITIAL
# =========================
def constructive_schedule(sessions, slots):
    schedule = {}
    used = defaultdict(set)

    for s in sessions:
        placed = False
        random.shuffle(slots)

        for i in range(len(slots)):
            ok = True
            block = slots[i:i + s["length"]]

            if len(block) < s["length"]:
                continue

            # harus satu hari & berurutan
            day = block[0][0]
            for j in range(len(block)):
                if block[j][0] != day:
                    ok = False
                if j > 0 and block[j][1] != block[j-1][1] + 1:
                    ok = False

            if not ok:
                continue

            for b in block:
                if b in used[s["kelas"]] or (b, s["guru"]) in used["guru"]:
                    ok = False

            if ok:
                for b in block:
                    schedule[(s["sid"], b)] = True
                    used[s["kelas"]].add(b)
                    used["guru"].add((b, s["guru"]))
                placed = True
                break

        if not placed:
            # fallback (HARUS tetap diisi)
            b = slots[0]
            schedule[(s["sid"], b)] = True
            used[s["kelas"]].add(b)
            used["guru"].add((b, s["guru"]))

    return schedule

# =========================
# FITNESS
# =========================
def fitness(schedule, sessions, slots):
    penalty = 0

    kelas_time = defaultdict(set)
    guru_time = defaultdict(set)

    for (sid, slot) in schedule:
        s = sessions[sid]
        if slot in kelas_time[s["kelas"]]:
            penalty += 5000
        if slot in guru_time[s["guru"]]:
            penalty += 5000

        kelas_time[s["kelas"]].add(slot)
        guru_time[s["guru"]].add(slot)

    # jam kosong penalty
    for k in kelas_time:
        by_day = defaultdict(list)
        for d,j in kelas_time[k]:
            by_day[d].append(j)
        for d in by_day:
            js = sorted(by_day[d])
            for i in range(1, len(js)):
                if js[i] != js[i-1] + 1:
                    penalty += 200

    return -penalty

# =========================
# MUTATION (SAFE)
# =========================
def mutate(schedule, sessions, slots):
    if random.random() > MUTATION_RATE:
        return schedule

    s = random.choice(sessions)
    sid = s["sid"]

    # hapus lama
    schedule = {k:v for k,v in schedule.items() if k[0] != sid}

    # taruh ulang
    for i in range(len(slots)):
        block = slots[i:i+s["length"]]
        if len(block) < s["length"]:
            continue

        day = block[0][0]
        ok = True
        for j in range(len(block)):
            if block[j][0] != day:
                ok = False
            if j > 0 and block[j][1] != block[j-1][1] + 1:
                ok = False
        if not ok:
            continue

        for b in block:
            schedule[(sid, b)] = True
        break

    return schedule

# =========================
# SAVE GRID
# =========================
def save_grid(schedule, sessions, slots, outdir):
    os.makedirs(outdir, exist_ok=True)

    grid = defaultdict(lambda: defaultdict(dict))

    for (sid, (d,j)) in schedule:
        s = sessions[sid]
        grid[s["kelas"]][d][j] = f"{s['mapel']}"

    for kelas in grid:
        path = os.path.join(outdir, f"{kelas}.csv")
        with open(path, "w", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(["Hari", "Jam", kelas])
            for d,j in slots:
                writer.writerow([d, j, grid[kelas][d].get(j, "")])

# =========================
# MAIN GA
# =========================
def run():
    tables = read_tables()
    sessions = build_sessions(tables)
    slots = build_slots(tables)

    print(f"[info] sessions: {len(sessions)}")
    print(f"[info] slots: {len(slots)}")

    population = [constructive_schedule(sessions, slots) for _ in range(POP_SIZE)]

    for gen in range(1, GENERATIONS+1):
        scored = [(fitness(p, sessions, slots), p) for p in population]
        scored.sort(reverse=True, key=lambda x: x[0])

        print(f"[Gen {gen:03d}] avg={sum(s for s,_ in scored)/len(scored):.2f} best={scored[0][0]:.2f}")

        new_pop = [p for _,p in scored[:ELITISM]]

        while len(new_pop) < POP_SIZE:
            parent = random.choice(scored[:10])[1]
            child = mutate(dict(parent), sessions, slots)
            new_pop.append(child)

        population = new_pop

    outdir = f"results/final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_grid(scored[0][1], sessions, slots, outdir)
    print(f"[DONE] saved to {outdir}")

# =========================
if __name__ == "__main__":
    run()
