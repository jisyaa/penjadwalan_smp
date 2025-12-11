#!/usr/bin/env python3
"""
ga_constraint_aware.py

Constraint-aware Genetic Algorithm for SMPN 1 Enam Lingkung (slot-sharing).
- Constructive initialization (heuristic) to build high-quality seeds
- Chunk-safe representation (sessions are blocks)
- Smart mutation (repair moves / swaps) and chunk-safe crossover
- Fitness with strong penalties for hard violations but graded rewards so GA can improve
- Saves each generation CSV and final CSV in results/run_<ts>/

Usage: python ga_constraint_aware.py
"""
import os, random, json, time
from datetime import datetime
from collections import defaultdict, namedtuple, Counter
from copy import deepcopy
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import trange

load_dotenv()

# ---------------- CONFIG ----------------
DB_USER = os.getenv("DB_USER","root")
DB_PASS = os.getenv("DB_PASS","")
DB_HOST = os.getenv("DB_HOST","127.0.0.1")
DB_PORT = os.getenv("DB_PORT","3306")
DB_NAME = os.getenv("DB_NAME","db_penjadwalan")

RESULTS_ROOT = os.getenv("RESULTS_DIR","./results")
POP_SIZE = int(os.getenv("POP_SIZE","40"))
GENERATIONS = int(os.getenv("GENERATIONS","120"))
RANDOM_SEED = int(os.getenv("SEED","12345"))

CONN_STR = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONN_STR, pool_pre_ping=True)

random.seed(RANDOM_SEED)

# ---------------- TYPES ----------------
Session = namedtuple("Session", ["sid","id_kelas","nama_kelas","id_mapel","nama_mapel","length","split_group"])

# ---------------- DB read helpers ----------------
def read_tables():
    with engine.connect() as conn:
        tables = {}
        for t in ["guru","mapel","kelas","guru_mapel","waktu"]:
            tables[t] = pd.read_sql(text(f"SELECT * FROM {t}"), conn)
    return tables

def build_slots(waktu_df):
    df = waktu_df.copy()
    avail = df[(df["jam_ke"].notnull()) & ((df["keterangan"].isnull()) | (df["keterangan"].astype(str).str.strip()==""))]
    avail = avail.sort_values(["hari","jam_ke"])
    id_to_slot = {int(r["id_waktu"]): r for _, r in avail.iterrows()}
    day_ordered = defaultdict(list)
    for _, r in avail.iterrows():
        day_ordered[r["hari"]].append(int(r["id_waktu"]))
    for d in day_ordered:
        day_ordered[d].sort(key=lambda wid: id_to_slot[wid]["jam_ke"])
    day_order = ["Senin","Selasa","Rabu","Kamis","Jumat"]
    ordered_slots = []
    for d in day_order:
        if d in day_ordered:
            ordered_slots.extend(day_ordered[d])
    return id_to_slot, day_ordered, ordered_slots

# ---------------- Create sessions according to jam_per_minggu mapel rules ----------------
def build_sessions(tables, target_hours_per_class=40):
    gm = tables["guru_mapel"][tables["guru_mapel"]["aktif"]=="aktif"]
    mapel_df = tables["mapel"]
    kelas_df = tables["kelas"]
    sessions=[]
    sid=1
    split_gid=1
    expected=defaultdict(int)
    pairs = gm.groupby(["id_kelas","id_mapel"]).size().reset_index()[["id_kelas","id_mapel"]]
    for _, r in pairs.iterrows():
        id_kelas = int(r["id_kelas"]); id_mapel = int(r["id_mapel"])
        kelas_row = kelas_df[kelas_df["id_kelas"]==id_kelas].iloc[0]
        mapel_row = mapel_df[mapel_df["id_mapel"]==id_mapel].iloc[0]
        nama_kelas = kelas_row["nama_kelas"]; nama_mapel = mapel_row["nama_mapel"]
        jam = int(mapel_row["jam_per_minggu"])
        if jam==1:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,1,None)); sid+=1
        elif jam==2:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,2,None)); sid+=1
        elif jam==3:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,3,None)); sid+=1
        elif jam==4:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,2,split_gid)); sid+=1
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,2,split_gid)); sid+=1
            split_gid+=1
        elif jam==5:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,3,split_gid)); sid+=1
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,2,split_gid)); sid+=1
            split_gid+=1
        elif jam==6:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,3,split_gid)); sid+=1
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,3,split_gid)); sid+=1
            split_gid+=1
        else:
            for _ in range(jam):
                sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,1,None)); sid+=1
        expected[nama_kelas] += jam
    # enforce consistent expected hours per class (user confirmed 40)
    for k in expected:
        expected[k] = target_hours_per_class
    return sessions, expected

def build_candidates(guru_mapel_df):
    cand = defaultdict(list)
    for _, r in guru_mapel_df[guru_mapel_df["aktif"]=="aktif"].iterrows():
        cand[(int(r["id_kelas"]), int(r["id_mapel"]))].append(int(r["id_guru"]))
    return cand

# ---------------- Utilities ----------------
def find_consecutive(start_wid, length, id_to_slot, day_ordered):
    if start_wid not in id_to_slot:
        return None
    day = id_to_slot[start_wid]["hari"]
    ordered = day_ordered[day]
    try:
        idx = ordered.index(start_wid)
    except ValueError:
        return None
    block = ordered[idx: idx+length]
    if len(block) != length:
        return None
    prev = None
    for wid in block:
        jk = id_to_slot[wid]["jam_ke"]
        if prev is not None and jk != prev + 1:
            return None
        prev = jk
    return block

# ---------------- Constructive initializer ----------------
def constructive_initial_individual(sessions, id_to_slot, day_ordered, candidates, ordered_class_names):
    """
    Place sessions class-by-class with heuristics:
    - place long sessions first (3h), try to allocate in days with available consecutive slots
    - avoid teacher collisions while building
    - attempt split_group sessions on different days
    """
    slots_map = {}
    guru_map = {}
    # busy map per teacher: (teacher, wid) -> True
    busy = {}
    # assigned per class: set of wid
    class_assigned = defaultdict(set)
    # track used slots per (class,day) to avoid creating gaps later by greedy approach
    per_class_day = defaultdict(lambda: defaultdict(int))

    # group sessions per class, sort classes to balance (shuffle)
    per_class = defaultdict(list)
    for s in sessions:
        per_class[s.nama_kelas].append(s)
    classes = ordered_class_names.copy()
    random.shuffle(classes)

    # place sessions per class
    for cls in classes:
        slist = sorted(per_class[cls], key=lambda x: (-x.length, x.split_group or 0))
        # prefer to fill days with more free consecutive capacity
        for s in slist:
            placed = False
            # build list of candidate (day,start,guru) sorted by "fit" score
            candidates_list = []
            for d, ordered in day_ordered.items():
                for i in range(len(ordered)):
                    start = ordered[i]
                    block = find_consecutive(start, s.length, id_to_slot, day_ordered)
                    if not block:
                        continue
                    # ensure block doesn't overlap existing class assignment
                    if any(w in class_assigned[cls] for w in block):
                        continue
                    # candidate teachers for this (class,mapel)
                    cands = candidates.get((s.id_kelas, s.id_mapel), [])
                    for g in cands:
                        # check quick teacher conflict on those wid
                        conflict = False
                        for wid in block:
                            if busy.get((g,wid), False):
                                conflict = True; break
                        if conflict:
                            continue
                        # heuristic score: prefer days with less assigned for this class to avoid gaps
                        score = -per_class_day[cls][d] - (0 if s.length==1 else 0)
                        candidates_list.append((score, d, start, g))
            # sort candidates by score ascending (better first)
            if candidates_list:
                candidates_list.sort(key=lambda x: x[0])
                for score,d,start,g in candidates_list:
                    block = find_consecutive(start, s.length, id_to_slot, day_ordered)
                    if not block: continue
                    # assign
                    slots_map[s.sid] = block[0]
                    guru_map[s.sid] = g
                    for wid in block:
                        busy[(g,wid)] = True
                        class_assigned[cls].add(wid)
                    per_class_day[cls][d] += s.length
                    placed = True
                    break
            if not placed:
                slots_map[s.sid] = None
                guru_map[s.sid] = None
    return slots_map, guru_map

def create_initial_population(sessions, id_to_slot, day_ordered, candidates, pop_size):
    class_names = sorted({s.nama_kelas for s in sessions})
    pop=[]
    # one deterministic greedy seed + randomized greedy variants
    pop.append(constructive_initial_individual(sessions, id_to_slot, day_ordered, candidates, class_names))
    for i in range(1, pop_size):
        order = class_names.copy()
        random.shuffle(order)
        pop.append(constructive_initial_individual(sessions, id_to_slot, day_ordered, candidates, order))
    return pop

# ---------------- Fitness (graded but heavy penalty for hard constraints) ----------------
def fitness_score(individual, sessions, id_to_slot, day_ordered, expected_per_class, candidates, guru_capacity):
    """
    Returns (score, diagnostics)
    Score higher is better. Hard violations reduce score sharply.
    We'll produce a numeric score that guides GA up even if not fully valid.
    """
    slots_map, guru_map = individual
    teacher_time = defaultdict(list)
    class_time = defaultdict(list)
    assigned_per_class = defaultdict(set)
    split_days = defaultdict(set)
    diag = {"missing":0,"non_consec":0,"blocked_slot":0,"teacher_conflict":0,"class_conflict":0,"split_violation":0,"gap":0,"expected_mismatch":0,"teacher_overload":0}

    # base score: each successfully assigned hour gives +1
    score = 0.0

    # Validate assignments and accumulate
    for s in sessions:
        st = slots_map.get(s.sid)
        g = guru_map.get(s.sid)
        if st is None or g is None:
            diag["missing"] += 1
            continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block:
            diag["non_consec"] += 1
            continue
        # blocked slots check
        blocked = False
        for wid in block:
            row = id_to_slot.get(wid)
            if row is None:
                blocked = True; break
            if row["keterangan"] is not None and str(row["keterangan"]).strip()!="":
                blocked = True; break
        if blocked:
            diag["blocked_slot"] += 1
            continue
        # register
        for wid in block:
            teacher_time[(g,wid)].append(s.sid)
            class_time[(s.nama_kelas,wid)].append(s.sid)
            assigned_per_class[s.nama_kelas].add(wid)
        if s.split_group is not None:
            split_days[s.split_group].add(id_to_slot[block[0]]["hari"])
        score += s.length  # rewarded per hour successfully placed

    # Teacher conflicts and class conflicts (heavy penalty)
    for (g,w), lst in teacher_time.items():
        if len(lst) > 1:
            # heavy penalty proportional to extra assignments
            extra = len(lst)-1
            diag["teacher_conflict"] += extra
            score -= 1000 * extra

    for (cls,w), lst in class_time.items():
        if len(lst) > 1:
            extra = len(lst)-1
            diag["class_conflict"] += extra
            score -= 1000 * extra

    # split groups must be on different days
    for gid, days in split_days.items():
        if len(days) < 2:
            diag["split_violation"] += 1
            score -= 800

    # gaps per class per day
    for cls, wids in assigned_per_class.items():
        per_day = defaultdict(list)
        for wid in wids:
            per_day[id_to_slot[wid]["hari"]].append(id_to_slot[wid]["jam_ke"])
        for day, jks in per_day.items():
            jks_sorted = sorted(jks)
            for i in range(len(jks_sorted)-1):
                if jks_sorted[i+1] != jks_sorted[i] + 1:
                    diag["gap"] += 1
                    score -= 300

    # expected hours per class exactness
    for cls, expected in expected_per_class.items():
        assigned = len(assigned_per_class.get(cls, set()))
        if assigned != expected:
            diff = abs(expected - assigned)
            diag["expected_mismatch"] += diff
            # penalize missing heavier than extra
            if assigned < expected:
                score -= 500 * diff
            else:
                score -= 100 * diff

    # teacher capacity
    teacher_hours = defaultdict(int)
    for (g,w), lst in teacher_time.items():
        teacher_hours[g] += len(lst)
    for g,h in teacher_hours.items():
        cap = guru_capacity.get(g, 9999)
        if h > cap:
            over = h - cap
            diag["teacher_overload"] += over
            score -= 500 * over

    # small bonuses to guide search:
    # - reward contiguous placement per class-day (lower variance)
    bonus = 0.0
    for cls, wids in assigned_per_class.items():
        per_day = defaultdict(int)
        for wid in wids:
            per_day[id_to_slot[wid]["hari"]] += 1
        vals = list(per_day.values())
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            bonus += max(0, 5.0 - var/5.0)
    score += bonus

    # final: if score very negative, clip to -1e9
    if score < -1e9:
        score = -1e9

    return score, diag

# ---------------- Operators ----------------
def tournament(pop, fitnesses, k=3):
    idxs = random.sample(range(len(pop)), min(k, len(pop)))
    best = max(idxs, key=lambda i: fitnesses[i])
    return deepcopy(pop[best])

def chunk_safe_crossover(p1, p2, sessions, prob=0.85):
    if random.random() > prob:
        return deepcopy(p1), deepcopy(p2)
    a_slots, a_gurus = deepcopy(p1[0]), deepcopy(p1[1])
    b_slots, b_gurus = deepcopy(p2[0]), deepcopy(p2[1])
    # swap per class-block groups to preserve locality
    # build map sid->class for grouping
    cls_map = defaultdict(list)
    for s in sessions:
        cls_map[s.nama_kelas].append(s.sid)
    # for each class, with 50% chance swap all its sessions as block
    for cls, sid_list in cls_map.items():
        if random.random() < 0.5:
            for sid in sid_list:
                a_slots[sid], b_slots[sid] = b_slots.get(sid), a_slots.get(sid)
                a_gurus[sid], b_gurus[sid] = b_gurus.get(sid), a_gurus.get(sid)
    return (a_slots, a_gurus), (b_slots, b_gurus)

def smart_mutation(ind, sessions, id_to_slot, day_ordered, candidates, prob_move=0.18, prob_swap=0.08):
    slots_map, guru_map = deepcopy(ind[0]), deepcopy(ind[1])
    # move: try to relocate problematic sessions to random valid starts
    problematic = []
    for s in sessions:
        st = slots_map.get(s.sid); g = guru_map.get(s.sid)
        if st is None or g is None or not find_consecutive(st, s.length, id_to_slot, day_ordered):
            problematic.append(s)
    # try to fix some problematic sessions by trying alternative starts/gurus
    for s in random.sample(problematic, min(len(problematic), max(1,len(problematic)//4))):
        if random.random() < prob_move:
            tries = []
            for d, ordered in day_ordered.items():
                for i in range(len(ordered)):
                    st = ordered[i]
                    if find_consecutive(st, s.length, id_to_slot, day_ordered):
                        tries.append(st)
            random.shuffle(tries)
            cands = candidates.get((s.id_kelas, s.id_mapel), [])
            if not cands: continue
            for st in tries[:40]:
                for g in random.sample(cands, min(len(cands),5)):
                    # simple accept (we don't check global teacher clash here; fitness will penalize)
                    slots_map[s.sid] = st
                    guru_map[s.sid] = g
                    if find_consecutive(st, s.length, id_to_slot, day_ordered):
                        # small immediate check: prevent obvious class overlap
                        conflict = False
                        for wid in find_consecutive(st, s.length, id_to_slot, day_ordered):
                            # check if another session of same class is at that wid
                            for ss in sessions:
                                if ss.nama_kelas == s.nama_kelas and ss.sid != s.sid:
                                    other_st = slots_map.get(ss.sid)
                                    if other_st is None: continue
                                    other_block = find_consecutive(other_st, ss.length, id_to_slot, day_ordered)
                                    if other_block and (wid in other_block):
                                        conflict = True; break
                            if conflict: break
                        if not conflict:
                            # accept this move
                            break
                else:
                    continue
                break
    # swap: swap sessions between two classes for same length to reduce conflicts
    if random.random() < prob_swap:
        # pick two sessions of same length
        s1, s2 = None, None
        bylen = defaultdict(list)
        for s in sessions:
            bylen[s.length].append(s)
        lengths = [L for L in bylen if len(bylen[L])>=2]
        if lengths:
            L = random.choice(lengths)
            s1, s2 = random.sample(bylen[L], 2)
            # swap their (slot,guru) if it doesn't obviously break adjacency (we will let fitness decide)
            tmp1 = slots_map.get(s1.sid); tmpg1 = guru_map.get(s1.sid)
            tmp2 = slots_map.get(s2.sid); tmpg2 = guru_map.get(s2.sid)
            slots_map[s1.sid], guru_map[s1.sid] = tmp2, tmpg2
            slots_map[s2.sid], guru_map[s2.sid] = tmp1, tmpg1
    return slots_map, guru_map

# ---------------- Repair local (attempt to improve individual before evaluation) ----------------
def repair_local(ind, sessions, id_to_slot, day_ordered, candidates, max_iter=200):
    slots_map, guru_map = deepcopy(ind[0]), deepcopy(ind[1])
    # try to fill missing sessions greedily while checking class overlap
    per_class_assigned = defaultdict(set)
    for s in sessions:
        st = slots_map.get(s.sid); g = guru_map.get(s.sid)
        if st is None or g is None:
            continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if block:
            for wid in block:
                per_class_assigned[s.nama_kelas].add(wid)
    # fill unassigned sessions
    unassigned = [s for s in sessions if slots_map.get(s.sid) is None or not find_consecutive(slots_map.get(s.sid, -1), s.length, id_to_slot, day_ordered)]
    random.shuffle(unassigned)
    for s in unassigned:
        placed = False
        tries = []
        for d, ordered in day_ordered.items():
            for i in range(len(ordered)):
                st = ordered[i]
                if find_consecutive(st, s.length, id_to_slot, day_ordered):
                    tries.append(st)
        random.shuffle(tries)
        cands = candidates.get((s.id_kelas, s.id_mapel), [])
        for st in tries:
            block = find_consecutive(st, s.length, id_to_slot, day_ordered)
            if not block: continue
            # avoid overlapping with same class
            if any(w in per_class_assigned[s.nama_kelas] for w in block):
                continue
            # try candidate teachers
            for g in random.sample(cands, min(len(cands),5)):
                # assign tentatively
                slots_map[s.sid] = st
                guru_map[s.sid] = g
                for wid in block:
                    per_class_assigned[s.nama_kelas].add(wid)
                placed = True
                break
            if placed: break
        if not placed:
            # leave None
            slots_map[s.sid] = None; guru_map[s.sid] = None
    return slots_map, guru_map

# ---------------- Save generation/final ----------------
    slots_map, guru_map = best_ind
def save_grid(run_dir, gen_idx, chrom, sessions, id_to_slot, ordered_slots, day_ordered, tables, tag="gen"):
    sub = f"{tag}_{gen_idx:03d}" if tag=="gen" else "final"
    dirp = os.path.join(run_dir, sub) if tag=="gen" else os.path.join(run_dir, "final")
    os.makedirs(dirp, exist_ok=True)

    slots_map, guru_map = chrom  # <-- ini yang benar

    rows=[]
    for wid in ordered_slots:
        r = id_to_slot[wid]
        rows.append({"id_waktu":wid,"hari":r["hari"],"jam_ke":r["jam_ke"]})
    df_slots = pd.DataFrame(rows)

    class_names = sorted({s.nama_kelas for s in sessions})
    table = pd.DataFrame(index=range(len(df_slots)), columns=["hari","jam_ke"]+class_names)
    table["hari"]=df_slots["hari"]
    table["jam_ke"]=df_slots["jam_ke"]
    table.fillna("", inplace=True)

    guru_df = tables["guru"]

    for s in sessions:
        st = slots_map.get(s.sid); g = guru_map.get(s.sid)
        if st is None or g is None:
            continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block:
            continue
        try:
            nama_g = guru_df.loc[guru_df["id_guru"]==g,"nama_guru"].iloc[0]
        except:
            nama_g = str(g)
        entry = f"{nama_g} - {s.nama_mapel}"
        for wid in block:
            idx = df_slots.index[df_slots["id_waktu"]==wid].tolist()
            if not idx: continue
            r = idx[0]
            table.at[r, s.nama_kelas] = entry

    fname = os.path.join(dirp, "combined_timetable.csv")
    table.to_csv(fname, index=False)

    return fname, table

# ---------------- MAIN GA LOOP ----------------
def run_ga():
    print("Constraint-aware GA starting...")
    tables = read_tables()
    id_to_slot, day_ordered, ordered_slots = build_slots(tables["waktu"])
    sessions, expected_per_class = build_sessions(tables, target_hours_per_class=40)
    candidates = build_candidates(tables["guru_mapel"])
    guru_capacity = {int(r["id_guru"]): int(r["jam_mingguan"]) for _, r in tables["guru"].iterrows()}

    print(f"[info] sessions count: {len(sessions)}")
    print(f"[info] slots template per week: {len(ordered_slots)} (reused per class)")
    # quick check: ensure candidates exist for each session
    missing_candidates = []
    for s in sessions:
        if len(candidates.get((s.id_kelas,s.id_mapel), [])) == 0:
            missing_candidates.append((s.nama_kelas, s.nama_mapel))
    if missing_candidates:
        print("[WARN] Some sessions have no candidate teacher (guru_mapel). Examples (class,mapel):")
        print(missing_candidates[:20])
        print("Please ensure guru_mapel contains active teachers for every (class,mapel). GA may not find valid solution.")
    # create results dir
    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT, f"run_{run_label}")
    os.makedirs(run_dir, exist_ok=True)
    # snapshot data
    for k,v in tables.items():
        try: v.to_csv(os.path.join(run_dir, f"{k}.csv"), index=False)
        except: pass

    # create initial population (constructive)
    population = create_initial_population(sessions, id_to_slot, day_ordered, candidates, POP_SIZE)
    # local repair on each member to improve feasibility
    population = [repair_local(ind, sessions, id_to_slot, day_ordered, candidates, max_iter=300) for ind in population]

    # evaluate initial
    fitnesses = []
    diags = []
    for ind in population:
        f, d = fitness_score(ind, sessions, id_to_slot, day_ordered, expected_per_class, candidates, guru_capacity)
        fitnesses.append(f); diags.append(d)
    print(f"[info] init avg={sum(fitnesses)/len(fitnesses):.6f} best={max(fitnesses):.6f}")

    best_overall = None; best_f_overall = -1e12
    # GA main loop
    for gen in range(1, GENERATIONS+1):
        avg = sum(fitnesses) / len(fitnesses)
        best_f = max(fitnesses); worst_f = min(fitnesses)
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        print(f"[Gen {gen:03d}] avg={avg:.6f} best={best_f:.6f} worst={worst_f:.6f}")
        # save generation best
        save_grid(run_dir, gen, population[best_idx], sessions, id_to_slot, ordered_slots, day_ordered, tables, tag="gen")
        if best_f > best_f_overall:
            best_f_overall = best_f; best_overall = deepcopy(population[best_idx])
        # selection + variation
        new_pop = [deepcopy(population[best_idx])]  # elitism 1
        while len(new_pop) < POP_SIZE:
            p1 = tournament(population, fitnesses, k=3)
            p2 = tournament(population, fitnesses, k=3)
            c1, c2 = chunk_safe_crossover(p1, p2, sessions, prob=0.85)
            c1 = smart_mutation(c1, sessions, id_to_slot, day_ordered, candidates, prob_move=0.18, prob_swap=0.08)
            c2 = smart_mutation(c2, sessions, id_to_slot, day_ordered, candidates, prob_move=0.18, prob_swap=0.08)
            c1 = repair_local(c1, sessions, id_to_slot, day_ordered, candidates, max_iter=200)
            c2 = repair_local(c2, sessions, id_to_slot, day_ordered, candidates, max_iter=200)
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)
        population = new_pop
        # evaluate
        fitnesses = []
        diags = []
        for ind in population:
            f, d = fitness_score(ind, sessions, id_to_slot, day_ordered, expected_per_class, candidates, guru_capacity)
            fitnesses.append(f); diags.append(d)
        # optional early stop if perfect
        if best_f_overall >= (40 * len(set([s.nama_kelas for s in sessions]))):  # heuristic perfect score
            print("[INFO] Achieved target heuristic score, stopping early.")
            break

    # save final best
    final_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    final_fname, final_table = save_grid(run_dir, 0, population[final_idx], sessions, id_to_slot, ordered_slots, day_ordered, tables, tag="final")
    print(f"[INFO] final saved to {os.path.dirname(final_fname)}")
    print(f"[INFO] best overall score: {best_f_overall:.6f}")
    # print sample diagnostics from best
    best_score, best_diag = fitness_score(population[final_idx], sessions, id_to_slot, day_ordered, expected_per_class, candidates, guru_capacity)
    print("[INFO] best final diagnostics sample:", best_diag)
    # print top part of final table for quick check
    with pd.option_context('display.max_rows', 200, 'display.max_columns', None):
        print(final_table.fillna("").head(40).to_string(index=False))
    # ask to save to DB
    ans = input("Simpan final ke database jadwal_master/jadwal? (y/n): ").strip().lower()
    if ans == "y":
        slots_map, guru_map = population[final_idx]
        tahun = input("Tahun ajaran (contoh: 2025/2026): ").strip()
        semester = input("Semester (ganjil/genap): ").strip()
        keterangan = input("Keterangan (opsional): ").strip()
        with engine.begin() as conn:
            res = conn.execute(text("INSERT INTO jadwal_master (tahun_ajaran, semester, keterangan, dibuat_pada) VALUES (:t,:s,:k,CURRENT_TIMESTAMP())"),
                               {"t":tahun,"s":semester,"k":keterangan})
            master_id = res.lastrowid
            rows=[]
            for s in sessions:
                st = slots_map.get(s.sid); g = guru_map.get(s.sid)
                if st is None or g is None: continue
                block = find_consecutive(st, s.length, id_to_slot, day_ordered)
                if not block: continue
                for wid in block:
                    rows.append({"id_master":master_id,"id_kelas":s.id_kelas,"id_mapel":s.id_mapel,"id_guru":g,"id_ruang":None,"id_waktu":wid,"generasi":None,"fitness":None})
            if rows:
                df = pd.DataFrame(rows)
                df.to_sql("jadwal", con=conn, if_exists="append", index=False)
        print(f"[DB] saved jadwal_master id {master_id} with {len(rows)} rows.")
    print("Done.")

if __name__ == "__main__":
    run_ga()
