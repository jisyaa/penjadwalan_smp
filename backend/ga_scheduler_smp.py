#!/usr/bin/env python3
"""
ga_scheduler_smp.py

Algoritma Genetik untuk penjadwalan mata pelajaran SMP (SMPN 1 Enam Lingkung).
Memenuhi constraint:
a. Satu guru tidak boleh mengajar di dua kelas pada waktu yang sama.
b. Setiap kelas harus memenuhi total jam pelajaran sesuai kurikulum.
c. Tidak boleh ada jam kosong di tengah hari belajar.
d. Waktu dengan keterangan != NULL tidak boleh diisi.
e. Gunakan hanya guru_mapel yang 'aktif'.
f. Tidak memasukkan ruangan (kelas punya ruang tetap).
g,h. Mapel dengan jam_per_minggu > 3 dibagi menjadi 2 pertemuan di hari berbeda,
     pertemuan yang panjang ditempatkan pada jam berdekatan.
i. Semua jam terisi kecuali yang ber-keterangan; tidak ada gap dan tidak tumpang tindih.

Fitur:
- Menampilkan fitness setiap generasi.
- Menampilkan jadwal generasi terbaik di terminal.
- Menanyakan apakah jadwal terbaik ingin disimpan ke DB (jadwal_master + jadwal).
- Menyimpan setiap generasi ke folder results/run_<timestamp>/gen_<n> (CSV).
- Mendukung banyak run (masing-masing run mendapat folder sendiri).

Author: Assistant (adapted for your dataset)
"""

import os
import random
import json
import math
import shutil
from datetime import datetime
from collections import defaultdict, namedtuple
from copy import deepcopy
from tqdm import tqdm

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ------------------ Konfigurasi DB & GA ------------------
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "db_penjadwalan")

CONN_STR = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONN_STR, pool_pre_ping=True)

RESULTS_ROOT = os.getenv("RESULTS_DIR", "./results")
POP_SIZE = int(os.getenv("POP_SIZE", 50))
GENERATIONS = int(os.getenv("GENERATIONS", 100))

random.seed(42)

# ------------------ Struktur data ------------------
Session = namedtuple("Session", ["id", "id_kelas", "nama_kelas", "id_mapel", "nama_mapel", "length", "split_group"])
# split_group: None or group id to enforce different days for split parts

# ------------------ Utility DB read ------------------
def read_tables():
    with engine.connect() as conn:
        tables = {}
        for name in ["guru", "mapel", "kelas", "guru_mapel", "waktu"]:
            tables[name] = pd.read_sql(text(f"SELECT * FROM {name}"), conn)
    return tables

# ------------------ Preprocessing sessions ------------------
def build_sessions(tables):
    """
    Convert guru_mapel + mapel + kelas into Session items.
    Rules:
    - Use only guru_mapel where aktif='aktif'.
    - For each (id_kelas, id_mapel) create sessions based on jam_per_minggu.
      If jam_per_minggu > 3 -> split into two pertemuan (lengths as balanced parts)
        and they must be on different days (we'll mark split_group).
      Else create jam_per_minggu sessions length=1.
    - For multi-hour pertemuan (length>1) we require consecutive waktu slots.
    """
    gm = tables["guru_mapel"][tables["guru_mapel"]["aktif"] == "aktif"]
    mapel = tables["mapel"].set_index("id_mapel")
    kelas = tables["kelas"].set_index("id_kelas")

    sessions = []
    sid = 1
    split_gid = 1

    # group gm by (id_kelas, id_mapel) to build sessions per class-mapel
    grouped = gm.groupby(["id_kelas", "id_mapel"]).size().reset_index()[["id_kelas", "id_mapel"]]
    for _, row in grouped.iterrows():
        id_kelas = int(row["id_kelas"])
        id_mapel = int(row["id_mapel"])
        nama_kelas = kelas.loc[id_kelas, "nama_kelas"]
        nama_mapel = mapel.loc[id_mapel, "nama_mapel"]
        jam = int(mapel.loc[id_mapel, "jam_per_minggu"])

        if jam > 3:
            # split into two meetings (lengths balanced) and days must differ
            a = math.ceil(jam/2)
            b = jam - a
            # ensure a>=b and both >0
            lengths = [a, b] if b>0 else [a]
            # if a>3 we still accept multi-hour meeting length a (but must find consecutive slots)
            # create two sessions with same split_group id
            for L in lengths:
                sessions.append(Session(sid, id_kelas, nama_kelas, id_mapel, nama_mapel, L, split_gid))
                sid += 1
            split_gid += 1
        else:
            # create jam sessions of length 1 repeated jam times
            for _ in range(jam):
                sessions.append(Session(sid, id_kelas, nama_kelas, id_mapel, nama_mapel, 1, None))
                sid += 1

    # sanity
    print(f"[preproc] Total sessions built: {len(sessions)}")
    return sessions

# ------------------ Time-slot helpers ------------------
def build_time_slots_table(tables):
    """
    Return list of waktu rows that are available (keterangan is null/empty and jam_ke not null).
    Also build mapping id_waktu -> (hari, jam_ke, waktu_mulai, waktu_selesai)
    """
    waktu = tables["waktu"].copy()
    # Exclude rows where keterangan not null/empty
    available = waktu[(waktu["keterangan"].isnull()) | (waktu["keterangan"].astype(str).str.strip()=="")]
    # Also require jam_ke not null (slots with NULL are breaks)
    available = available[available["jam_ke"].notnull()]
    available = available.sort_values(["hari","jam_ke"])
    slots = available.to_dict("records")
    id_to_slot = {int(r["id_waktu"]): r for r in slots}
    # For adjacent check we need order per (hari, jam_ke)
    # also map day->ordered list of id_waktu
    day_slots = defaultdict(list)
    for r in slots:
        day_slots[r["hari"]].append(int(r["id_waktu"]))
    # sort each day by jam_ke
    for d in day_slots:
        day_slots[d].sort(key=lambda wid: id_to_slot[wid]["jam_ke"])
    return id_to_slot, day_slots

# ------------------ Candidate lists (guru) ------------------
def build_candidates(tables):
    """
    For each (id_kelas, id_mapel) get list of guru candidates (from guru_mapel aktif)
    Return dict keyed by (id_kelas, id_mapel) -> list of id_guru
    Also return guru jam_mingguan capacity
    """
    gm = tables["guru_mapel"][tables["guru_mapel"]["aktif"] == "aktif"]
    # assume gm contains id_kelas, id_mapel, id_guru
    cand = defaultdict(list)
    for _, row in gm.iterrows():
        cand[(int(row["id_kelas"]), int(row["id_mapel"]))].append(int(row["id_guru"]))
    # guru capacities (jam_mingguan)
    guru_df = tables["guru"].set_index("id_guru")
    guru_capacity = {int(idx): int(guru_df.loc[idx, "jam_mingguan"]) for idx in guru_df.index}
    return cand, guru_capacity

# ------------------ Chromosome representation ------------------
# Chromosome: dict { session.id : assigned_id_waktu } with implied assignment to class fixed
# For sessions with length>1 we store start_id_waktu; consumer must expand to consecutive slots
# Additionally we store assigned_guru for each session: dict {session.id: guru_id}
# We'll encode chromosome as two dicts: slots_map (session->start_slot), guru_map (session->guru)

# ------------------ Feasibility utilities ------------------
def find_consecutive_slots(start_wid, length, id_to_slot, day_slots):
    """Check if from start_wid there are `length` consecutive slots on same day.
       Return list of wids or None."""
    if start_wid not in id_to_slot:
        return None
    day = id_to_slot[start_wid]["hari"]
    ordered = day_slots[day]
    # positions
    try:
        idx = ordered.index(start_wid)
    except ValueError:
        return None
    needed = ordered[idx: idx+length]
    if len(needed) != length:
        return None
    # check continuity in jam_ke (ensured by day_slots order)
    return needed

def all_available_start_slots_for_length(id_to_slot, day_slots, length):
    """Return list of start_wid such that length consecutive slots exist."""
    starts = []
    for day, ordered in day_slots.items():
        for i in range(len(ordered)):
            if i + length <= len(ordered):
                starts.append(ordered[i])
    return starts

# ------------------ Building initial population ------------------
def generate_initial_population(sessions, id_to_slot, day_slots, candidates_map, pop_size=POP_SIZE):
    """
    Generate population of chromosomes.
    Each chromosome: (slots_map, guru_map)
    - For sessions length>1: pick a random day that has consecutive length slots
      and ensure for split sessions day differs from other split part by group id (we will attempt)
    - Ensure no assignments to forbidden slots (we only use available slots)
    """
    # precompute possible starts per length grouped by day
    length_to_starts = {}
    for s in sessions:
        L = s.length
        if L not in length_to_starts:
            starts = []
            for day, ordered in day_slots.items():
                for i in range(len(ordered)):
                    if i + L <= len(ordered):
                        starts.append(ordered[i])
            length_to_starts[L] = starts

    population = []
    for _ in range(pop_size):
        slots_map = {}
        guru_map = {}
        # track split_group assigned days to ensure different days for same group
        split_group_days = {}
        for s in sessions:
            L = s.length
            starts = length_to_starts.get(L, [])
            if not starts:
                # no possible start -> leave None (will be penalized)
                start = None
            else:
                # if split group constraint exists try ensure different day
                if s.split_group is not None and s.split_group in split_group_days:
                    # pick start that is on a different day than existing
                    existing_day = split_group_days[s.split_group]
                    # filter starts
                    filtered = [st for st in starts if id_to_slot[st]["hari"] != existing_day]
                    if filtered:
                        start = random.choice(filtered)
                    else:
                        start = random.choice(starts)
                else:
                    start = random.choice(starts)
                if s.split_group is not None:
                    split_group_days[s.split_group] = id_to_slot[start]["hari"]
            slots_map[s.id] = start
            # choose a random candidate guru for the class-mapel
            key = (s.id_kelas, s.id_mapel)
            cand = candidates_map.get(key, [])
            if cand:
                guru_map[s.id] = random.choice(cand)
            else:
                # fallback: random from all gurus (should be rare)
                guru_map[s.id] = random.choice(list(candidates_map.values())[0]) if candidates_map else None
        population.append((slots_map, guru_map))
    return population

# ------------------ Fitness function (key) ------------------
def evaluate_fitness(chromosome, sessions, id_to_slot, day_slots, candidates_map, guru_capacity):
    """
    chromosome: (slots_map, guru_map)
    Return fitness score (higher better) and diagnostics dict
    Penalty scheme (example weights):
     - teacher conflict: 100 per conflict
     - missing assignment (start None or no consecutive slots): 200 per session
     - gap per class (empty slot between assigned slots in a day across all days): 5 per gap
     - split-in-same-day violation: 50 per violation
     - teacher overload (assigned hours > capacity): 20 per extra hour
    We will compute raw_penalty and convert to fitness = 1/(1+penalty)
    """
    slots_map, guru_map = chromosome
    penalty = 0
    details = {"teacher_conflicts": [], "missing": [], "gaps": 0, "split_violation": [], "teacher_overload": {}}

    # Build expanded assignment maps:
    # teacher_time[(guru, id_waktu)] -> list session ids
    teacher_time = defaultdict(list)
    # class_time[(kelas, id_waktu)] -> list session ids
    class_time = defaultdict(list)
    # assigned_slots_per_class[kelas] = set of id_waktu assigned
    assigned_slots_per_class = defaultdict(set)
    # track hours per guru
    guru_hours = defaultdict(int)
    # track split_group days
    split_days = defaultdict(set)

    # Expand sessions into their consecutive slots and validate existence
    for s in sessions:
        start = slots_map.get(s.id)
        if start is None:
            penalty += 200
            details["missing"].append((s.id, "no_start"))
            continue
        consec = find_consecutive_slots(start, s.length, id_to_slot, day_slots)
        if not consec:
            penalty += 200
            details["missing"].append((s.id, "no_consecutive"))
            continue
        # check each slot in consec: add teacher_time/class_time
        g = guru_map.get(s.id)
        for wid in consec:
            teacher_time[(g, wid)].append(s.id)
            class_time[(s.id_kelas, wid)].append(s.id)
            assigned_slots_per_class[s.id_kelas].add(wid)
            guru_hours[g] += 1
        # record split group day
        if s.split_group is not None:
            split_days[s.split_group].add(id_to_slot[start]["hari"])

    # a) teacher conflicts: teacher_time entries with >1
    t_conf_count = 0
    for (g, wid), sids in teacher_time.items():
        if len(sids) > 1:
            t_conf_count += 1
            penalty += 100 * (len(sids)-1)  # heavier if more sessions
            details["teacher_conflicts"].append((g, wid, sids))
    # b) each class must meet total jam per minggu: check by summing sessions lengths for that class
    # compute expected per class from sessions
    expected_per_class = defaultdict(int)
    for s in sessions:
        expected_per_class[s.id_kelas] += s.length
    for kelas, expected in expected_per_class.items():
        assigned = len(assigned_slots_per_class.get(kelas, set()))
        if assigned < expected:
            # missing slots for class
            diff = expected - assigned
            penalty += 200 * diff
            details["missing"].append(("class_missing", kelas, diff))
        elif assigned > expected:
            # over-assigned (should not happen but penalize)
            diff = assigned - expected
            penalty += 50 * diff
            details["missing"].append(("class_over", kelas, diff))
    # c) no gap mid-day for each class:
    # For each class and day, get sorted jam_ke indices and count gaps (holes)
    for kelas, slots in assigned_slots_per_class.items():
        # build per day slots
        per_day = defaultdict(list)
        for wid in slots:
            day = id_to_slot[wid]["hari"]
            per_day[day].append(id_to_slot[wid]["jam_ke"])
        for day, jks in per_day.items():
            jks_sorted = sorted(jks)
            # if single slot, no gap; else gaps = number of missing jam_ke in between
            for i in range(len(jks_sorted)-1):
                if jks_sorted[i+1] != jks_sorted[i] + 1:
                    # there's a gap
                    penalty += 5
                    details["gaps"] += 1
    # d) times with keterangan not null we excluded from available; ensure none scheduled there
    # (already prevented by using only available slots)
    # e) only aktif used (we built candidates_map from aktif). check if assigned guru is in candidates:
    for s in sessions:
        g = guru_map.get(s.id)
        key = (s.id_kelas, s.id_mapel)
        cands = candidates_map.get(key, [])
        if g not in cands:
            # heavy penalty
            penalty += 500
            details.setdefault("invalid_assignment", []).append((s.id, g))
    # g,h) split sessions: ensure split group sessions are on different days
    for group, days in split_days.items():
        if len(days) < 2:
            # violation: both parts on same day or one missing
            penalty += 50
            details["split_violation"].append((group, list(days)))
    # teacher overload: guru_hours > guru_capacity
    for g, hours in guru_hours.items():
        cap = guru_capacity.get(g, 9999)
        if hours > cap:
            extra = hours - cap
            penalty += 20 * extra
            details["teacher_overload"][g] = extra

    # final fitness
    fitness = 1.0 / (1.0 + penalty)
    diagnostics = {"penalty": penalty, **details}
    return fitness, diagnostics

# ------------------ Genetic operators ------------------
def tournament_selection(population, fitnesses, k=3):
    """Return selected individual (slots_map, guru_map) using tournament selection."""
    selected = random.sample(list(range(len(population))), k)
    best = max(selected, key=lambda i: fitnesses[i])
    return deepcopy(population[best])

def crossover(parent1, parent2, sessions, prob_crossover=0.8):
    """One-point crossover on session id list: swap a subset of sessions assignments (both slot & guru)."""
    if random.random() > prob_crossover:
        return deepcopy(parent1), deepcopy(parent2)
    s_ids = [s.id for s in sessions]
    a = random.randint(0, len(s_ids)-1)
    b = random.randint(a, len(s_ids)-1)
    child1_slots = deepcopy(parent1[0])
    child1_guru = deepcopy(parent1[1])
    child2_slots = deepcopy(parent2[0])
    child2_guru = deepcopy(parent2[1])
    for sid in s_ids[a:b+1]:
        child1_slots[sid], child2_slots[sid] = child2_slots.get(sid), child1_slots.get(sid)
        child1_guru[sid], child2_guru[sid] = child2_guru.get(sid), child1_guru.get(sid)
    return (child1_slots, child1_guru), (child2_slots, child2_guru)

def mutation(individual, sessions, id_to_slot, day_slots, candidates_map, prob_mutation=0.1):
    """Mutate by reassigning random sessions' start slot and/or guru."""
    slots_map, guru_map = deepcopy(individual[0]), deepcopy(individual[1])
    for s in sessions:
        if random.random() < prob_mutation:
            # reassign start slot: pick random valid start for the session length
            L = s.length
            starts = []
            # prepare starts: any day with enough consecutive slots
            for day, ordered in day_slots.items():
                for i in range(len(ordered)):
                    if i + L <= len(ordered):
                        starts.append(ordered[i])
            if starts:
                slots_map[s.id] = random.choice(starts)
            # reassign guru from candidates
            key = (s.id_kelas, s.id_mapel)
            cands = candidates_map.get(key, [])
            if cands:
                guru_map[s.id] = random.choice(cands)
    return slots_map, guru_map

# ------------------ I/O: save generation results ------------------
def save_generation_result(run_dir, gen_idx, population, fitnesses, sessions, id_to_slot):
    """
    Save best individual's schedule CSV and a summary JSON for the generation.
    Also save CSV of best schedule with expanded slots.
    """
    gen_dir = os.path.join(run_dir, f"gen_{gen_idx:03d}")
    os.makedirs(gen_dir, exist_ok=True)
    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    best = population[best_idx]
    best_fitness = fitnesses[best_idx]

    # expand best into rows: session id, kelas, mapel, guru, day, jam_ke (for each slot)
    rows = []
    for s in sessions:
        start = best[0].get(s.id)
        g = best[1].get(s.id)
        if start is None:
            rows.append({
                "id_session": s.id, "kelas": s.nama_kelas, "mapel": s.nama_mapel,
                "guru": g, "start_waktu": None, "length": s.length, "detail": "MISSING"
            })
            continue
        consec = find_consecutive_slots(start, s.length, id_to_slot, day_slots)
        if not consec:
            rows.append({
                "id_session": s.id, "kelas": s.nama_kelas, "mapel": s.nama_mapel,
                "guru": g, "start_waktu": start, "length": s.length, "detail": "INVALID_CONSEC"
            })
            continue
        for wid in consec:
            rows.append({
                "id_session": s.id, "kelas": s.nama_kelas, "mapel": s.nama_mapel,
                "guru": g, "id_waktu": wid, "hari": id_to_slot[wid]["hari"],
                "jam_ke": id_to_slot[wid]["jam_ke"]
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(gen_dir, "best_schedule.csv"), index=False)
    summary = {
        "gen": gen_idx,
        "best_fitness": float(best_fitness),
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(gen_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # save population fitness list
    pd.DataFrame({"fitness": fitnesses}).to_csv(os.path.join(gen_dir, "fitnesses.csv"), index=False)
    return best_idx, best_fitness, gen_dir

# ------------------ Save final best to DB ------------------
def save_best_to_db(best_individual, sessions, id_to_slot, run_label=None):
    """
    Ask user details for jadwal_master then insert jadwal_master row and jadwal rows
    """
    slots_map, guru_map = best_individual
    # ask user
    tahun = input("Masukkan Tahun Ajaran (mis: 2025/2026): ").strip()
    semester = input("Masukkan Semester (ganjil/genap): ").strip().lower()
    keterangan = input("Keterangan tambahan (opsional): ").strip()
    created_at = datetime.now()

    # insert master
    with engine.begin() as conn:
        res = conn.execute(text("INSERT INTO jadwal_master (tahun_ajaran, semester, keterangan, dibuat_pada) VALUES (:t, :s, :k, :d)"),
                           {"t": tahun, "s": semester, "k": keterangan, "d": created_at})
        master_id = res.lastrowid
        print(f"[db] Created jadwal_master id = {master_id}")
        # insert jadwal rows: expand each session into its consecutive slot rows
        rows = []
        for s in sessions:
            start = slots_map.get(s.id)
            g = guru_map.get(s.id)
            if start is None:
                continue
            consec = find_consecutive_slots(start, s.length, id_to_slot, day_slots)
            if not consec:
                continue
            for wid in consec:
                rows.append({
                    "id_master": master_id,
                    "id_kelas": s.id_kelas,
                    "id_mapel": s.id_mapel,
                    "id_guru": g,
                    "id_ruang": None,  # not used
                    "id_waktu": wid,
                    "generasi": 0,
                    "fitness": None
                })
        if rows:
            df = pd.DataFrame(rows)
            # insert batch
            df.to_sql("jadwal", con=conn, if_exists="append", index=False)
            print(f"[db] Inserted {len(rows)} jadwal rows into jadwal table.")
    return master_id

# ------------------ Main GA loop ------------------
def run_ga(pop_size=POP_SIZE, generations=GENERATIONS, prob_crossover=0.8, prob_mutation=0.1):
    tables = read_tables()
    global candidates_map, guru_capacity, id_to_slot, day_slots
    candidates_map, guru_capacity = build_candidates(tables)
    sessions = build_sessions(tables)
    id_to_slot, day_slots = build_time_slots_table(tables)

    # quick check: ensure sum expected_per_class <= available slots per class across week
    # compute available slots per day for each class (classes use fixed id_ruang but we only need count)
    available_slots_count = sum(len(v) for v in day_slots.values())
    total_required_slots = sum(s.length for s in sessions)
    print(f"[info] total required slots (sum session lengths): {total_required_slots}")
    print(f"[info] total available slots in week: {available_slots_count * len(day_slots)} (per day counted)")  # approximate

    # create run folder
    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT, f"run_{run_label}")
    os.makedirs(run_dir, exist_ok=True)
    # also copy a snapshot of DB tables for record
    for tname, df in tables.items():
        try:
            df.to_csv(os.path.join(run_dir, f"{tname}.csv"), index=False)
        except Exception:
            pass

    # initial population
    population = generate_initial_population(sessions, id_to_slot, day_slots, candidates_map, pop_size=pop_size)
    fitness_history = []
    best_overall = None
    best_fitness_overall = -1
    # GA loop
    for gen in range(1, generations+1):
        fitnesses = []
        diagnostics_list = []
        for ind in population:
            f, diag = evaluate_fitness(ind, sessions, id_to_slot, day_slots, candidates_map, guru_capacity)
            fitnesses.append(f)
            diagnostics_list.append(diag)

        # log gen stats
        avg_f = sum(fitnesses)/len(fitnesses)
        max_f = max(fitnesses)
        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        print(f"[Gen {gen:03d}] avg fitness = {avg_f:.6f}, best fitness = {max_f:.6f}")
        # save generation results
        best_idx, best_f, gen_dir = save_generation_result(run_dir, gen, population, fitnesses, sessions, id_to_slot)
        fitness_history.append({"gen": gen, "avg": avg_f, "best": max_f, "gen_dir": gen_dir})
        # update overall best
        if max_f > best_fitness_overall:
            best_fitness_overall = max_f
            best_overall = deepcopy(population[best_idx])

        # create next generation
        new_pop = []
        # elitism: keep top 1
        sorted_idx = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        elite = deepcopy(population[sorted_idx[0]])
        new_pop.append(elite)
        while len(new_pop) < pop_size:
            # selection
            p1 = tournament_selection(population, fitnesses, k=3)
            p2 = tournament_selection(population, fitnesses, k=3)
            # crossover
            c1, c2 = crossover(p1, p2, sessions, prob_crossover=prob_crossover)
            # mutation
            c1 = mutation(c1, sessions, id_to_slot, day_slots, candidates_map, prob_mutation)
            c2 = mutation(c2, sessions, id_to_slot, day_slots, candidates_map, prob_mutation)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        population = new_pop

    # after GA
    print(f"=== GA complete. Best fitness overall = {best_fitness_overall:.6f} ===")
    # show best schedule (expanded)
    best_slots_map, best_guru_map = best_overall
    rows = []
    for s in sessions:
        start = best_slots_map.get(s.id)
        g = best_guru_map.get(s.id)
        if start is None:
            rows.append({"session": s.id, "kelas": s.nama_kelas, "mapel": s.nama_mapel, "guru": g, "hari": None, "jam_ke": None, "note": "MISSING"})
            continue
        consec = find_consecutive_slots(start, s.length, id_to_slot, day_slots)
        if not consec:
            rows.append({"session": s.id, "kelas": s.nama_kelas, "mapel": s.nama_mapel, "guru": g, "hari": None, "jam_ke": None, "note": "INVALID"})
            continue
        for wid in consec:
            rows.append({"session": s.id, "kelas": s.nama_kelas, "mapel": s.nama_mapel, "guru": g,
                         "hari": id_to_slot[wid]["hari"], "jam_ke": id_to_slot[wid]["jam_ke"]})
    df_best = pd.DataFrame(rows)
    print(df_best.sort_values(["kelas","hari","jam_ke"]).to_string(index=False))
    # ask user to save to DB
    saveq = input("Apakah ingin menyimpan jadwal terbaik ke database master/jadwal? (y/n): ").strip().lower()
    if saveq == "y":
        master_id = save_best_to_db(best_overall, sessions, id_to_slot, run_label)
        print(f"[DONE] saved to jadwal_master id {master_id}")
    else:
        print("[INFO] Not saved to DB.")
    print(f"[INFO] All generation results saved under: {run_dir}")
    return run_dir

# ------------------ Run script ------------------
if __name__ == "__main__":
    print("Starting GA scheduler...")
    run_dir = run_ga()
    print("Finished.")
