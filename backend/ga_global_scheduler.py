#!/usr/bin/env python3
"""
ga_global_scheduler.py

Global Genetic Algorithm scheduler for SMPN 1 Enam Lingkung.
1 individual = full-week schedule for all classes (global timetable).

Features:
- Constraints: teacher double-book, class fills expected hours, no gap mid-day, no-slot-with-keterangan,
  split sessions as specified (3->3,4->2+2,5->3+2,6->3+3), consecutive blocks, only aktif guru_mapel used.
- Save every generation to results/run_<timestamp>/gen_<nnn>/combined_timetable.csv and fitness.json.
- Print avg/best/worst fitness per generation.
- Ask to save best schedule to DB (jadwal_master + jadwal).

Note: This script can be heavy. Start with small POP_SIZE and GENERATIONS.
"""

import os
import random
import json
import math
from datetime import datetime
from collections import defaultdict, namedtuple, Counter
from copy import deepcopy
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import trange

load_dotenv()

# ---------- Config ----------
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "db_penjadwalan")
RESULTS_ROOT = os.getenv("RESULTS_DIR", "./results")
POP_SIZE = int(os.getenv("POP_SIZE", 30))
GENERATIONS = int(os.getenv("GENERATIONS", 60))

CONN_STR = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONN_STR, pool_pre_ping=True)

random.seed(42)

# ---------- Data structures ----------
Session = namedtuple("Session", ["sid","id_kelas","nama_kelas","id_mapel","nama_mapel","length","split_group"])

# ---------- DB read ----------
def read_tables():
    with engine.connect() as conn:
        tables = {}
        for t in ["guru","mapel","kelas","guru_mapel","waktu"]:
            tables[t] = pd.read_sql(text(f"SELECT * FROM {t}"), conn)
    return tables

# ---------- Build available time slots ----------
def build_slots(waktu_df):
    df = waktu_df.copy()
    # available slots: keterangan null or empty, jam_ke not null
    avail = df[(df["keterangan"].isnull()) | (df["keterangan"].astype(str).str.strip()=="")]
    avail = avail[avail["jam_ke"].notnull()]
    avail = avail.sort_values(["hari","jam_ke"])
    id_to_slot = {int(r["id_waktu"]): r for _, r in avail.iterrows()}
    # day -> ordered list of id_waktu
    day_ordered = defaultdict(list)
    for _, r in avail.iterrows():
        day_ordered[r["hari"]].append(int(r["id_waktu"]))
    # sort each day by jam_ke
    for d in day_ordered:
        day_ordered[d].sort(key=lambda wid: id_to_slot[wid]["jam_ke"])
    # create global ordered slots list (by day then jam)
    ordered_slots = []
    day_order = ["Senin","Selasa","Rabu","Kamis","Jumat"]
    for day in day_order:
        if day in day_ordered:
            ordered_slots.extend(day_ordered[day])
    return id_to_slot, day_ordered, ordered_slots

# ---------- Build sessions for all classes (global) ----------
def build_all_sessions(tables):
    """
    For each class, for each mapel assigned (guru_mapel aktif), create sessions using rules:
    3 -> [3]
    4 -> [2,2]
    5 -> [3,2]
    6 -> [3,3]
    others -> 1 x jam
    Return sessions list and expected_per_class dict.
    """
    gm = tables["guru_mapel"][tables["guru_mapel"]["aktif"]=="aktif"]
    mapel_df = tables["mapel"]
    kelas_df = tables["kelas"]
    sessions = []
    sid = 1
    split_gid = 1
    expected_per_class = defaultdict(int)
    # group by class and mapel
    pairs = gm.groupby(["id_kelas","id_mapel"]).size().reset_index()[["id_kelas","id_mapel"]]
    for _, row in pairs.iterrows():
        id_kelas = int(row["id_kelas"])
        id_mapel = int(row["id_mapel"])
        kelas_row = kelas_df[kelas_df["id_kelas"]==id_kelas].iloc[0]
        mapel_row = mapel_df[mapel_df["id_mapel"]==id_mapel].iloc[0]
        nama_kelas = kelas_row["nama_kelas"]
        nama_mapel = mapel_row["nama_mapel"]
        jam = int(mapel_row["jam_per_minggu"])
        if jam == 3:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,3,None)); sid+=1
        elif jam == 4:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,2,split_gid)); sid+=1
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,2,split_gid)); sid+=1
            split_gid += 1
        elif jam == 5:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,3,split_gid)); sid+=1
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,2,split_gid)); sid+=1
            split_gid += 1
        elif jam == 6:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,3,split_gid)); sid+=1
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,3,split_gid)); sid+=1
            split_gid += 1
        else:
            for _ in range(jam):
                sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,1,None)); sid+=1
        expected_per_class[nama_kelas] += jam
    # expected_per_class now contains jam_per_minggu sums
    return sessions, expected_per_class

# ---------- Candidate map ----------
def build_candidates_map(guru_mapel_df):
    cand = defaultdict(list)  # (id_kelas,id_mapel)->list id_guru
    for _, r in guru_mapel_df[guru_mapel_df["aktif"]=="aktif"].iterrows():
        cand[(int(r["id_kelas"]), int(r["id_mapel"]))].append(int(r["id_guru"]))
    return cand

# ---------- Utilities: consecutive slot check ----------
def find_consecutive(start_wid, length, id_to_slot, day_ordered):
    if start_wid not in id_to_slot: return None
    day = id_to_slot[start_wid]["hari"]
    ordered = day_ordered[day]
    try:
        idx = ordered.index(start_wid)
    except ValueError:
        return None
    block = ordered[idx: idx+length]
    if len(block) != length:
        return None
    # check jam_ke consecutive
    prev = None
    for wid in block:
        jk = id_to_slot[wid]["jam_ke"]
        if prev is not None and jk != prev + 1:
            return None
        prev = jk
    return block

# ---------- Chromosome representation ----------
# chromosome: dict { session.sid -> start_wid } and dict { session.sid -> id_guru }
# length>1 means start_wid represents consecutive block

# ---------- Greedy + randomized initial population (global) ----------
def create_initial_population_global(sessions, id_to_slot, day_ordered, ordered_slots, candidates_map, global_teacher_busy, pop_size):
    """
    Attempt to build pop_size individuals.
    We'll use greedy packing per class in order, with some randomness so individuals differ.
    """
    # Build sessions grouped per class to pack without gaps
    class_sessions = defaultdict(list)
    for s in sessions:
        class_sessions[s.nama_kelas].append(s)
    # ensure sessions per class sorted (longer first)
    for k in class_sessions:
        class_sessions[k].sort(key=lambda x: (-x.length, x.split_group or 0, x.id_mapel))
    population = []
    for _ in range(pop_size):
        slots_map = {}
        guru_map = {}
        # make a fresh copy of busy map
        busy = dict(global_teacher_busy)
        # process classes in random order
        classes = list(class_sessions.keys())
        random.shuffle(classes)
        for cls in classes:
            assigned_slots = set()
            split_used_days = {}
            sess_list = class_sessions[cls]
            # fill days in natural order but with shuffle to produce variety
            days = list(day_ordered.keys())
            # deterministic order: Senin..Jumat
            day_order = ["Senin","Selasa","Rabu","Kamis","Jumat"]
            days = [d for d in day_order if d in day_ordered]
            for s in sess_list:
                placed = False
                # try days; shuffle to create diversity but keep deterministic preference
                day_choices = days.copy()
                random.shuffle(day_choices)
                # If split_group already used day, prefer other days
                if s.split_group is not None and s.split_group in split_used_days:
                    # prefer days not equal
                    d_pref = [d for d in day_choices if d != split_used_days[s.split_group]] + [d for d in day_choices if d == split_used_days[s.split_group]]
                    day_choices = d_pref
                for d in day_choices:
                    ordered = day_ordered[d]
                    # try each possible start index
                    for i in range(len(ordered)):
                        start = ordered[i]
                        block = find_consecutive(start, s.length, id_to_slot, day_ordered)
                        if not block: continue
                        # skip if any block slot conflicts with assigned_slots
                        if any(w in assigned_slots for w in block): continue
                        # check teacher candidates
                        cands = candidates_map.get((s.id_kelas, s.id_mapel), [])
                        random.shuffle(cands)
                        for g in cands:
                            conflict=False
                            for wid in block:
                                if busy.get((g,wid), False):
                                    conflict=True; break
                            if not conflict:
                                # assign
                                slots_map[s.sid] = block[0]
                                guru_map[s.sid] = g
                                # mark busy
                                for wid in block:
                                    busy[(g,wid)] = True
                                    assigned_slots.add(wid)
                                if s.split_group is not None:
                                    split_used_days[s.split_group] = d
                                placed = True
                                break
                        if placed: break
                    if placed: break
                if not placed:
                    # leave unassigned for now (will be penalized / repaired later)
                    slots_map[s.sid] = None
                    guru_map[s.sid] = None
        population.append((slots_map, guru_map))
    return population

# ---------- Fitness (global) ----------
def evaluate_chromosome(chrom, sessions, id_to_slot, day_ordered, ordered_slots, expected_per_class, candidates_map, guru_capacity):
    """
    Returns fitness (higher better) and diagnostics.
    Strong penalties for:
      - teacher double-book (per conflicting slot) -> 1e6
      - class double-book (two sessions of same class in same wid) -> 1e6
      - missing assignment or non-consecutive block -> 1e6
      - split-group same day -> 1e5
      - gap per class per day -> 1000 per gap
      - teacher overload -> 500 per extra hour
      - total hours not meeting expected -> 1e5 per missing hour
    Fitness = 1 / (1 + penalty)
    """
    slots_map, guru_map = chrom
    penalty = 0
    details = {"teacher_conflicts":0,"class_conflicts":0,"missing":0,"gaps":0,"split_viol":0,"overload":0,"missing_hours":0}
    # build maps
    teacher_time = defaultdict(list)   # (guru,wid) -> [sid...]
    class_time = defaultdict(list)     # (kelas,wid) -> [sid...]
    assigned_per_class = defaultdict(set)
    split_days = defaultdict(set)
    # expand
    for s in sessions:
        start = slots_map.get(s.sid)
        g = guru_map.get(s.sid)
        if start is None or g is None:
            penalty += 1_000_00  # heavy
            details["missing"] += 1
            continue
        block = find_consecutive(start, s.length, id_to_slot, day_ordered)
        if not block:
            penalty += 1_000_00
            details["missing"] += 1
            continue
        for wid in block:
            teacher_time[(g,wid)].append(s.sid)
            class_time[(s.nama_kelas,wid)].append(s.sid)
            assigned_per_class[s.nama_kelas].add(wid)
        if s.split_group is not None:
            split_days[s.split_group].add(id_to_slot[block[0]]["hari"])
    # teacher conflicts
    for (g,wid), lst in teacher_time.items():
        if len(lst) > 1:
            # penalize heavily per extra booking
            count = len(lst)-1
            penalty += 1_000_000 * count
            details["teacher_conflicts"] += count
    # class conflicts
    for (cls,wid), lst in class_time.items():
        if len(lst) > 1:
            count = len(lst)-1
            penalty += 1_000_000 * count
            details["class_conflicts"] += count
    # gaps per class per day
    # for each class, group assigned wid by day and check missing holes within min..max
    for cls, wids in assigned_per_class.items():
        per_day = defaultdict(list)
        for wid in wids:
            day = id_to_slot[wid]["hari"]
            per_day[day].append(id_to_slot[wid]["jam_ke"])
        for day, jks in per_day.items():
            jks_sorted = sorted(jks)
            for i in range(len(jks_sorted)-1):
                if jks_sorted[i+1] != jks_sorted[i] + 1:
                    penalty += 1000
                    details["gaps"] += 1
    # split violation
    for gid, days in split_days.items():
        if len(days) < 2:
            penalty += 100_000
            details["split_viol"] += 1
    # expected hours per class
    for cls, expected in expected_per_class.items():
        assigned = len(assigned_per_class.get(cls, set()))
        if assigned < expected:
            diff = expected - assigned
            penalty += 100_000 * diff
            details["missing_hours"] += diff
        elif assigned > expected:
            # shouldn't happen but penalize
            penalty += 50_000 * (assigned - expected)
            details.setdefault("over_assign",0)
            details["over_assign"] += (assigned - expected)
    # teacher overload: compute teacher total assigned hours
    teacher_hours = defaultdict(int)
    for (g,wid), lst in teacher_time.items():
        teacher_hours[g] += len(lst)
    # capacity from guru_capacity
    for g, hours in teacher_hours.items():
        cap = guru_capacity.get(g, 9999)
        if hours > cap:
            extra = hours - cap
            penalty += 500 * extra
            details["overload"] += extra
    fitness = 1.0 / (1.0 + penalty)
    return fitness, penalty, details

# ---------- GA operators ----------
def tournament_select(pop, fitnesses, k=3):
    idxs = random.sample(range(len(pop)), min(k,len(pop)))
    best = max(idxs, key=lambda i: fitnesses[i])
    return deepcopy(pop[best])

def crossover(parent1, parent2, sessions, prob=0.85):
    if random.random() > prob:
        return deepcopy(parent1), deepcopy(parent2)
    # uniform crossover by session subset: choose random mask
    p1_slots, p1_gurus = deepcopy(parent1[0]), deepcopy(parent1[1])
    p2_slots, p2_gurus = deepcopy(parent2[0]), deepcopy(parent2[1])
    for s in sessions:
        if random.random() < 0.5:
            p1_slots[s.sid], p2_slots[s.sid] = p2_slots.get(s.sid), p1_slots.get(s.sid)
            p1_gurus[s.sid], p2_gurus[s.sid] = p2_gurus.get(s.sid), p1_gurus.get(s.sid)
    return (p1_slots,p1_gurus),(p2_slots,p2_gurus)

def mutation(individual, sessions, id_to_slot, day_ordered, candidates_map, prob_mut=0.12):
    slots_map, guru_map = deepcopy(individual[0]), deepcopy(individual[1])
    for s in sessions:
        if random.random() < prob_mut:
            # change start slot to random valid start for length
            starts = []
            for d, ordered in day_ordered.items():
                for i in range(len(ordered)):
                    start = ordered[i]
                    block = find_consecutive(start, s.length, id_to_slot, day_ordered)
                    if block: starts.append(start)
            if starts:
                slots_map[s.sid] = random.choice(starts)
        if random.random() < prob_mut:
            cands = candidates_map.get((s.id_kelas, s.id_mapel), [])
            if cands:
                guru_map[s.sid] = random.choice(cands)
    return (slots_map, guru_map)

# ---------- Repair (lightweight) ----------
def repair(individual, sessions, id_to_slot, day_ordered, candidates_map, global_teacher_busy):
    # Attempts to fix missing / conflicts locally:
    slots_map, guru_map = deepcopy(individual[0]), deepcopy(individual[1])
    # 1) Fill missing: try to place session in free block and available teacher
    # build quick teacher busy map from global_teacher_busy (only for previously fixed items)
    busy = dict(global_teacher_busy)
    # also consider current assignments to avoid class double-book
    class_assign = defaultdict(set)
    for s in sessions:
        st = slots_map.get(s.sid)
        g = guru_map.get(s.sid)
        if st is None or g is None: continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block: 
            slots_map[s.sid] = None; guru_map[s.sid] = None; continue
        conflict=False
        for wid in block:
            if wid in class_assign[s.nama_kelas]:
                conflict=True; break
            if busy.get((g,wid), False):
                conflict=True; break
        if conflict:
            # unassign so will be reattempted
            slots_map[s.sid] = None; guru_map[s.sid] = None
        else:
            for wid in block:
                class_assign[s.nama_kelas].add(wid)
                busy[(g,wid)] = True
    # Now try to place unassigned sessions
    sessions_sorted = sorted([s for s in sessions if slots_map.get(s.sid) is None], key=lambda x:-x.length)
    days = ["Senin","Selasa","Rabu","Kamis","Jumat"]
    for s in sessions_sorted:
        placed=False
        for d in days:
            ordered = day_ordered.get(d, [])
            for i in range(len(ordered)):
                start = ordered[i]
                block = find_consecutive(start, s.length, id_to_slot, day_ordered)
                if not block: continue
                # check class_assign
                if any(w in class_assign[s.nama_kelas] for w in block): continue
                # try candidate gurus
                cands = candidates_map.get((s.id_kelas,s.id_mapel), [])
                for g in cands:
                    conflict=False
                    for wid in block:
                        if busy.get((g,wid), False):
                            conflict=True; break
                    if not conflict:
                        slots_map[s.sid]=block[0]; guru_map[s.sid]=g
                        for wid in block:
                            class_assign[s.nama_kelas].add(wid)
                            busy[(g,wid)] = True
                        placed=True; break
                if placed: break
            if placed: break
    return (slots_map, guru_map)

# ---------- Save generation (global) ----------
def save_generation_global(run_dir, gen_idx, population, fitnesses, best_idx, sessions, id_to_slot, ordered_slots, tables):
    gen_dir = os.path.join(run_dir, f"gen_{gen_idx:03d}")
    os.makedirs(gen_dir, exist_ok=True)
    best = population[best_idx]
    best_f = fitnesses[best_idx]
    # expand best to combined timetable DataFrame (Format B)
    # build slots df
    rows=[]
    for wid in ordered_slots:
        r = id_to_slot[wid]
        rows.append({"id_waktu":wid,"hari":r["hari"],"jam_ke":r["jam_ke"]})
    df_slots = pd.DataFrame(rows)
    class_names = sorted({s.nama_kelas for s in sessions})
    table = pd.DataFrame(index=range(len(df_slots)), columns=["hari","jam_ke"]+class_names)
    table["hari"] = df_slots["hari"]; table["jam_ke"] = df_slots["jam_ke"]
    # fill with empty
    for c in class_names:
        table[c] = ""
    # fill using best assignments
    b_slots, b_gurus = best
    # make lookups for guru name & mapel name
    guru_df = tables["guru"]; mapel_df = tables["mapel"]
    for s in sessions:
        start = b_slots.get(s.sid)
        g = b_gurus.get(s.sid)
        if start is None or g is None: continue
        block = find_consecutive(start, s.length, id_to_slot, day_ordered)
        if not block: continue
        # guru name
        try:
            nama_g = guru_df.loc[guru_df["id_guru"]==g,"nama_guru"].iloc[0]
        except Exception:
            nama_g = str(g)
        for wid in block:
            idx = df_slots.index[df_slots["id_waktu"]==wid].tolist()
            if not idx: continue
            r = idx[0]
            current = table.at[r, s.nama_kelas]
            entry = f"{nama_g} - {s.nama_mapel}"
            # if cell empty, assign; else append (should not happen if no class conflicts)
            if current is None or str(current).strip()=="":
                table.at[r, s.nama_kelas] = entry
            else:
                table.at[r, s.nama_kelas] = current + " | " + entry
    # save CSV and fitness
    table.to_csv(os.path.join(gen_dir,"combined_timetable.csv"), index=False)
    with open(os.path.join(gen_dir,"fitness.json"),"w",encoding="utf-8") as f:
        json.dump({"best_fitness":float(best_f),"avg_fitness":float(sum(fitnesses)/len(fitnesses)),"worst":float(min(fitnesses))}, f, indent=2)
    return gen_dir, table

# ---------- Save best to DB ----------
def save_best_to_db(best_individual, sessions, id_to_slot, ordered_slots, tables):
    b_slots, b_gurus = best_individual
    # ask user
    tahun = input("Masukkan Tahun Ajaran (mis: 2025/2026): ").strip()
    semester = input("Masukkan Semester (ganjil/genap): ").strip()
    keterangan = input("Keterangan (opsional): ").strip()
    created_at = datetime.now()
    # insert master and jadwal
    with engine.begin() as conn:
        res = conn.execute(text("INSERT INTO jadwal_master (tahun_ajaran, semester, keterangan, dibuat_pada) VALUES (:t,:s,:k,:d)"),
                           {"t":tahun,"s":semester,"k":keterangan,"d":created_at})
        master_id = res.lastrowid
        # expand best and insert rows
        rows=[]
        for s in sessions:
            start = b_slots.get(s.sid)
            g = b_gurus.get(s.sid)
            if start is None or g is None: continue
            block = find_consecutive(start, s.length, id_to_slot, day_ordered)
            if not block: continue
            for wid in block:
                rows.append({"id_master":master_id,"id_kelas":s.id_kelas,"id_mapel":s.id_mapel,"id_guru":g,"id_ruang":None,"id_waktu":wid,"generasi":None,"fitness":None})
        if rows:
            df = pd.DataFrame(rows)
            df.to_sql("jadwal", con=conn, if_exists="append", index=False)
    print(f"[DB] Saved jadwal_master id = {master_id} and {len(rows)} rows to jadwal.")
    return master_id

# ---------- Main GA global ----------
if __name__ == "__main__":
    print("Starting GA Global scheduler...")
    tables = read_tables()
    id_to_slot, day_ordered, ordered_slots = build_slots(tables["waktu"])
    sessions, expected_per_class = build_all_sessions(tables)
    print(f"[info] Total sessions (sum of per-meeting blocks): {len(sessions)}")
    print(f"[info] Expected hours per class (sample): {dict(list(expected_per_class.items())[:5])}")
    candidates_map = build_candidates_map(tables["guru_mapel"])
    # guru capacity
    guru_capacity = {int(r["id_guru"]): int(r["jam_mingguan"]) for _,r in tables["guru"].iterrows()}

    # Quick check: total required slots vs available
    total_required = sum(val for val in expected_per_class.values())
    total_available = len(ordered_slots)
    # number of days * slots per day? for full week available slots count:
    print(f"[info] Total required slots (sum expected hours): {total_required}")
    print(f"[info] Total available slots in week (non-blocked): {len(ordered_slots)}")
    if total_required > len(ordered_slots):
        print("[WARN] Required slots exceed available slots. Algorithm will penalize/may not find feasible solution.")
        # but proceed

    # create run dir
    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT, f"run_{run_label}")
    os.makedirs(run_dir, exist_ok=True)
    # snapshot input tables
    for k,v in tables.items():
        try: v.to_csv(os.path.join(run_dir, f"{k}.csv"), index=False)
        except: pass

    # global teacher busy initially empty
    global_teacher_busy = {}  # (guru, wid) -> True if occupied
    # initial population
    print("[info] Generating initial population...")
    population = create_initial_population_global(sessions, id_to_slot, day_ordered, ordered_slots, candidates_map, global_teacher_busy, POP_SIZE)
    # evaluate initial
    fitnesses = []
    for ind in population:
        f,pen,diag = evaluate_chromosome(ind, sessions, id_to_slot, day_ordered, ordered_slots, expected_per_class, candidates_map, guru_capacity)
        fitnesses.append(f)
    print(f"[info] Initial population: avg fitness {sum(fitnesses)/len(fitnesses):.8f}, best {max(fitnesses):.8f}")

    best_global = None; best_global_f = -1
    # GA loop
    for gen in range(1, GENERATIONS+1):
        new_population = []
        new_fitnesses = []
        # elitism: keep top 2
        sorted_idx = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        elites = [deepcopy(population[sorted_idx[0]]), deepcopy(population[sorted_idx[1]])] if len(population)>1 else [deepcopy(population[sorted_idx[0]])]
        # compute stats
        avg_f = sum(fitnesses)/len(fitnesses)
        best_f = max(fitnesses)
        worst_f = min(fitnesses)
        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        print(f"[Gen {gen:03d}] avg={avg_f:.8f} best={best_f:.8f} worst={worst_f:.8f}")
        # save generation (before evolution step) results
        gen_dir, table = save_generation_global(run_dir, gen, population, fitnesses, best_idx, sessions, id_to_slot, ordered_slots, tables)
        # update global best
        if best_f > best_global_f:
            best_global_f = best_f
            best_global = deepcopy(population[best_idx])
        # next generation
        new_population.extend(elites)
        while len(new_population) < POP_SIZE:
            p1 = tournament_select(population, fitnesses, k=3)
            p2 = tournament_select(population, fitnesses, k=3)
            c1, c2 = crossover(p1, p2, sessions, prob=0.85)
            c1 = mutation(c1, sessions, id_to_slot, day_ordered, candidates_map, prob_mut=0.12)
            c2 = mutation(c2, sessions, id_to_slot, day_ordered, candidates_map, prob_mut=0.12)
            # repair using lightweight heuristic (global teacher busy not enforced across population here)
            c1 = repair(c1, sessions, id_to_slot, day_ordered, candidates_map, {})
            c2 = repair(c2, sessions, id_to_slot, day_ordered, candidates_map, {})
            new_population.append(c1)
            if len(new_population) < POP_SIZE:
                new_population.append(c2)
        # evaluate new population
        population = new_population
        fitnesses = []
        for ind in population:
            f,pen,diag = evaluate_chromosome(ind, sessions, id_to_slot, day_ordered, ordered_slots, expected_per_class, candidates_map, guru_capacity)
            fitnesses.append(f)

    # End GA
    print("=== GA finished ===")
    print(f"Best overall fitness = {best_global_f:.8f}")
    # save final best to run_dir/final_best.csv
    # expand best_global to combined table one more time
    final_best_idx = 0  # best_global stored separately
    final_gen_dir = os.path.join(run_dir,"final")
    os.makedirs(final_gen_dir, exist_ok=True)
    # reuse save_generation_global for best
    # temporarily create pop of [best_global]
    pop_dummy = [best_global]
    f_dummy = [best_global_f]
    # create sessions etc
    _, final_table = save_generation_global(run_dir, GENERATIONS+1, pop_dummy, f_dummy, 0, sessions, id_to_slot, ordered_slots, tables)
    print("[INFO] Final best timetable preview:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(final_table.fillna("").to_string(index=False))
    # ask save to DB
    ans = input("Simpan jadwal terbaik ke database jadwal_master/jadwal? (y/n): ").strip().lower()
    if ans == "y":
        master_id = save_best_to_db(best_global, sessions, id_to_slot, ordered_slots, tables)
        print(f"[DONE] saved to DB under jadwal_master id {master_id}")
    else:
        print("[INFO] Did not save to DB.")
    print(f"[INFO] All generation data saved under: {run_dir}")
    print("Finished.")
