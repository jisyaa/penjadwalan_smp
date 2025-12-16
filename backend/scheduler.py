#!/usr/bin/env python3
"""
ga_constraint_aware.py

Constraint-aware Genetic Algorithm for SMPN 1 Enam Lingkung (slot-sharing).
- Constructive initialization (heuristic) to build high-quality seeds
- Chunk-safe representation (sessions are blocks)
- Smart mutation (repair moves / swaps) and chunk-safe crossover
- Fitness with strong penalties for hard violations but graded rewards so GA can improve
- Enforces: session splitting rules (1,3,4,5,6), split-group on different days,
  and prevents same subject for same class on same day.
- Saves each generation CSV and final CSV in results/run_<ts>/

Usage: python ga_constraint_aware.py
"""
import os
import random
import time
from datetime import datetime
from collections import defaultdict, namedtuple
from copy import deepcopy
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "db_penjadwalan")

RESULTS_ROOT = os.getenv("RESULTS_DIR", "./results")
POP_SIZE = int(os.getenv("POP_SIZE", "40"))
GENERATIONS = int(os.getenv("GENERATIONS", "120"))
RANDOM_SEED = int(os.getenv("SEED", "12345"))

CONN_STR = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONN_STR, pool_pre_ping=True)

random.seed(RANDOM_SEED)

# ---------------- TYPES ----------------
Session = namedtuple(
    "Session", ["sid", "id_kelas", "nama_kelas", "id_mapel", "nama_mapel", "length", "split_group"]
)

# ---------------- DB read helpers ----------------
def read_tables():
    with engine.connect() as conn:
        tables = {}
        for t in ["guru", "mapel", "kelas", "guru_mapel", "waktu"]:
            tables[t] = pd.read_sql(text(f"SELECT * FROM {t}"), conn)
    return tables

def build_slots(waktu_df):
    df = waktu_df.copy()
    avail = df[
        (df["jam_ke"].notnull())
        & ((df["keterangan"].isnull()) | (df["keterangan"].astype(str).str.strip() == ""))
    ]
    avail = avail.sort_values(["hari", "jam_ke"])
    id_to_slot = {int(r["id_waktu"]): r for _, r in avail.iterrows()}
    day_ordered = defaultdict(list)
    for _, r in avail.iterrows():
        day_ordered[r["hari"]].append(int(r["id_waktu"]))
    for d in day_ordered:
        day_ordered[d].sort(key=lambda wid: id_to_slot[wid]["jam_ke"])
    day_order = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat"]
    ordered_slots = []
    for d in day_order:
        if d in day_ordered:
            ordered_slots.extend(day_ordered[d])
    return id_to_slot, day_ordered, ordered_slots

# ---------------- Create sessions according to jam_per_minggu mapel rules ----------------
def build_sessions(tables, target_hours_per_class=40):
    gm = tables["guru_mapel"][tables["guru_mapel"]["aktif"] == "aktif"]
    mapel_df = tables["mapel"]
    kelas_df = tables["kelas"]
    sessions = []
    sid = 1
    split_gid = 1
    expected = defaultdict(int)
    pairs = gm.groupby(["id_kelas", "id_mapel"]).size().reset_index()[["id_kelas", "id_mapel"]]
    for _, r in pairs.iterrows():
        id_kelas = int(r["id_kelas"])
        id_mapel = int(r["id_mapel"])
        kelas_row = kelas_df[kelas_df["id_kelas"] == id_kelas].iloc[0]
        mapel_row = mapel_df[mapel_df["id_mapel"] == id_mapel].iloc[0]
        nama_kelas = kelas_row["nama_kelas"]
        nama_mapel = mapel_row["nama_mapel"]
        jam = int(mapel_row["jam_per_minggu"])
        # follow rules provided by user
        if jam == 1:
            sessions.append(Session(sid, id_kelas, nama_kelas, id_mapel, nama_mapel, 1, None)); sid += 1
        elif jam == 2:
            sessions.append(Session(sid, id_kelas, nama_kelas, id_mapel, nama_mapel, 2, None)); sid += 1
        elif jam == 3:
            sessions.append(Session(sid, id_kelas, nama_kelas, id_mapel, nama_mapel, 3, None)); sid += 1
        elif jam == 4:
            sessions.append(Session(sid, id_kelas, nama_kelas, id_mapel, nama_mapel, 2, split_gid)); sid += 1
            sessions.append(Session(sid, id_kelas, nama_kelas, id_mapel, nama_mapel, 2, split_gid)); sid += 1
            split_gid += 1
        elif jam == 5:
            sessions.append(Session(sid, id_kelas, nama_kelas, id_mapel, nama_mapel, 3, split_gid)); sid += 1
            sessions.append(Session(sid, id_kelas, nama_kelas, id_mapel, nama_mapel, 2, split_gid)); sid += 1
            split_gid += 1
        elif jam == 6:
            sessions.append(Session(sid, id_kelas, nama_kelas, id_mapel, nama_mapel, 3, split_gid)); sid += 1
            sessions.append(Session(sid, id_kelas, nama_kelas, id_mapel, nama_mapel, 3, split_gid)); sid += 1
            split_gid += 1
        else:
            for _ in range(jam):
                sessions.append(Session(sid, id_kelas, nama_kelas, id_mapel, nama_mapel, 1, None)); sid += 1
        expected[nama_kelas] += jam
    # enforce consistent expected hours per class (user confirmed 40)
    for k in expected:
        expected[k] = target_hours_per_class
    return sessions, expected

def build_candidates(guru_mapel_df):
    cand = defaultdict(list)
    for _, r in guru_mapel_df[guru_mapel_df["aktif"] == "aktif"].iterrows():
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
    block = ordered[idx: idx + length]
    if len(block) != length:
        return None
    prev = None
    for wid in block:
        jk = id_to_slot[wid]["jam_ke"]
        if prev is not None and jk != prev + 1:
            return None
        prev = jk
    return block

# ---------------- Constructive initializer (with per-class per-mapel-day checks) ----------------
def constructive_initial_individual(sessions, id_to_slot, day_ordered, candidates, ordered_class_names):
    slots_map = {}
    guru_map = {}
    busy = {}  # (guru,wid)->True
    class_assigned = defaultdict(set)
    per_class_day = defaultdict(lambda: defaultdict(int))

    # new trackers to prevent same mapel twice a day & split group duplication per day
    per_class_mapel_day = defaultdict(lambda: defaultdict(int))  # per_class_mapel_day[class][(day,id_mapel)] = hours
    placed_split_days = defaultdict(set)  # placed_split_days[split_group] = set(days)

    # group sessions per class
    per_class = defaultdict(list)
    for s in sessions:
        per_class[s.nama_kelas].append(s)
    classes = ordered_class_names.copy()
    random.shuffle(classes)

    for cls in classes:
        slist = sorted(per_class[cls], key=lambda x: (-x.length, x.split_group or 0))
        for s in slist:
            placed = False
            candidates_list = []
            for d, ordered in day_ordered.items():
                for i in range(len(ordered)):
                    start = ordered[i]
                    block = find_consecutive(start, s.length, id_to_slot, day_ordered)
                    if not block:
                        continue
                    # avoid overlapping with already assigned slots for same class
                    if any(w in class_assigned[cls] for w in block):
                        continue
                    # prevent same mapel for same class on same day
                    if per_class_mapel_day[cls].get((d, s.id_mapel), 0) > 0:
                        continue
                    # prevent placing split_group twice same day
                    if s.split_group is not None and (d in placed_split_days[s.split_group]):
                        continue
                    cands = candidates.get((s.id_kelas, s.id_mapel), [])
                    for g in cands:
                        conflict = False
                        for wid in block:
                            if busy.get((g, wid), False):
                                conflict = True
                                break
                        if conflict:
                            continue
                        # heuristic score: prefer days with less assigned to avoid gaps
                        score = per_class_day[cls][d]
                        candidates_list.append((score, d, start, g))
            if candidates_list:
                candidates_list.sort(key=lambda x: x[0])
                for score, d, start, g in candidates_list:
                    block = find_consecutive(start, s.length, id_to_slot, day_ordered)
                    if not block:
                        continue
                    # assign
                    slots_map[s.sid] = block[0]
                    guru_map[s.sid] = g
                    for wid in block:
                        busy[(g, wid)] = True
                        class_assigned[cls].add(wid)
                    per_class_day[cls][d] += s.length
                    per_class_mapel_day[cls][(d, s.id_mapel)] += s.length
                    if s.split_group is not None:
                        placed_split_days[s.split_group].add(d)
                    placed = True
                    break
            if not placed:
                slots_map[s.sid] = None
                guru_map[s.sid] = None
    return slots_map, guru_map

def create_initial_population(sessions, id_to_slot, day_ordered, candidates, pop_size):
    class_names = sorted({s.nama_kelas for s in sessions})
    pop = []
    pop.append(constructive_initial_individual(sessions, id_to_slot, day_ordered, candidates, class_names))
    for i in range(1, pop_size):
        order = class_names.copy()
        random.shuffle(order)
        pop.append(constructive_initial_individual(sessions, id_to_slot, day_ordered, candidates, order))
    return pop

# ---------------- Fitness (graded but heavy penalty for hard constraints) ----------------
def fitness_score(individual, sessions, id_to_slot, day_ordered, expected_per_class, candidates, guru_capacity):
    slots_map, guru_map = individual
    teacher_time = defaultdict(list)
    class_time = defaultdict(list)
    assigned_per_class = defaultdict(set)
    split_days = defaultdict(set)
    diag = {
        "missing": 0,
        "non_consec": 0,
        "blocked_slot": 0,
        "teacher_conflict": 0,
        "class_conflict": 0,
        "split_violation": 0,
        "gap": 0,
        "expected_mismatch": 0,
        "teacher_overload": 0,
        "daily_subject_overload": 0,
    }

    score = 0.0

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
        blocked = False
        for wid in block:
            row = id_to_slot.get(wid)
            if row is None:
                blocked = True
                break
            if row["keterangan"] is not None and str(row["keterangan"]).strip() != "":
                blocked = True
                break
        if blocked:
            diag["blocked_slot"] += 1
            continue
        for wid in block:
            teacher_time[(g, wid)].append(s.sid)
            class_time[(s.nama_kelas, wid)].append(s.sid)
            assigned_per_class[s.nama_kelas].add(wid)
        if s.split_group is not None:
            split_days[s.split_group].add(id_to_slot[block[0]]["hari"])
        score += s.length

    for (g, w), lst in teacher_time.items():
        if len(lst) > 1:
            extra = len(lst) - 1
            diag["teacher_conflict"] += extra
            score -= 1000 * extra

    for (cls, w), lst in class_time.items():
        if len(lst) > 1:
            extra = len(lst) - 1
            diag["class_conflict"] += extra
            score -= 1000 * extra

    for gid, days in split_days.items():
        if len(days) < 2:
            diag["split_violation"] += 1
            score -= 800

    # gaps per class per day & collect per-subject-per-day totals
    per_day_subject = defaultdict(lambda: defaultdict(int))  # per_day_subject[class][day] = hours of same subject aggregated
    # build mapping sid->session to look up name and length quickly
    sid_map = {s.sid: s for s in sessions}
    for cls, wids in assigned_per_class.items():
        per_day = defaultdict(list)
        for wid in wids:
            per_day[id_to_slot[wid]["hari"]].append(id_to_slot[wid]["jam_ke"])
        for day, jks in per_day.items():
            jks_sorted = sorted(jks)
            for i in range(len(jks_sorted) - 1):
                if jks_sorted[i + 1] != jks_sorted[i] + 1:
                    diag["gap"] += 1
                    score -= 300
        # compute per subject-hours per day for this class
    # compute per-day per-subject totals by scanning assigned sessions
    for s in sessions:
        st = slots_map.get(s.sid)
        if st is None:
            continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block:
            continue
        d = id_to_slot[block[0]]["hari"]
        per_day_subject[s.nama_kelas][(d, s.id_mapel)] += s.length

    # if same mapel (id_mapel) appears > 1 time on same day for a class -> heavy penalty
    for cls, days in per_day_subject.items():
        # days keyed by (day,id_mapel); we want totals per (day, mapel)
        per_day_totals = defaultdict(int)
        for (d, mid), hrs in days.items():
            per_day_totals[(d, mid)] += hrs
        for (d, mid), total_len in per_day_totals.items():
            # The enforced rule: a single mapel for a class should not have two separate sessions on same day.
            # So if total_len > allowed_by_session (for jam 3, it would be 3 but split logic prevents two sessions same day).
            if total_len > 0:
                # check if this mapel is represented by more than one session on the same day:
                # count sessions of that mapel at that day
                cnt_sessions = 0
                for s in sessions:
                    st = slots_map.get(s.sid)
                    if not st:
                        continue
                    block = find_consecutive(st, s.length, id_to_slot, day_ordered)
                    if not block:
                        continue
                    if s.nama_kelas == cls and s.id_mapel == mid and id_to_slot[block[0]]["hari"] == d:
                        cnt_sessions += 1
                if cnt_sessions > 1:
                    excess = cnt_sessions - 1
                    diag["daily_subject_overload"] += excess
                    score -= 800 * excess

    # expected hours per class exactness
    for cls, expected in expected_per_class.items():
        assigned = len(assigned_per_class.get(cls, set()))
        if assigned != expected:
            diff = abs(expected - assigned)
            diag["expected_mismatch"] += diff
            if assigned < expected:
                score -= 500 * diff
            else:
                score -= 100 * diff

    # teacher capacity
    teacher_hours = defaultdict(int)
    for (g, w), lst in teacher_time.items():
        teacher_hours[g] += len(lst)
    for g, h in teacher_hours.items():
        cap = guru_capacity.get(g, 9999)
        if h > cap:
            over = h - cap
            diag["teacher_overload"] += over
            score -= 500 * over

    # small bonus for balanced days
    bonus = 0.0
    for cls, wids in assigned_per_class.items():
        per_day = defaultdict(int)
        for wid in wids:
            per_day[id_to_slot[wid]["hari"]] += 1
        vals = list(per_day.values())
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            bonus += max(0, 5.0 - var / 5.0)
    score += bonus

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
    cls_map = defaultdict(list)
    for s in sessions:
        cls_map[s.nama_kelas].append(s.sid)
    for cls, sid_list in cls_map.items():
        if random.random() < 0.5:
            for sid in sid_list:
                a_slots[sid], b_slots[sid] = b_slots.get(sid), a_slots.get(sid)
                a_gurus[sid], b_gurus[sid] = b_gurus.get(sid), a_gurus.get(sid)
    return (a_slots, a_gurus), (b_slots, b_gurus)

# ---------------- Smart mutation (with checks to avoid same-mapel same-day) ----------------
def smart_mutation(ind, sessions, id_to_slot, day_ordered, candidates, prob_move=0.18, prob_swap=0.08):
    slots_map, guru_map = deepcopy(ind[0]), deepcopy(ind[1])

    # compute existing per-class per-day mapel placed
    per_class_mapel_day = defaultdict(lambda: defaultdict(int))
    placed_split_days = defaultdict(set)
    for s in sessions:
        st = slots_map.get(s.sid)
        if st is None:
            continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block:
            continue
        d = id_to_slot[block[0]]["hari"]
        per_class_mapel_day[s.nama_kelas][(d, s.id_mapel)] += s.length
        if s.split_group is not None:
            placed_split_days[s.split_group].add(d)

    problematic = []
    for s in sessions:
        st = slots_map.get(s.sid)
        g = guru_map.get(s.sid)
        if st is None or g is None or not find_consecutive(st, s.length, id_to_slot, day_ordered):
            problematic.append(s)

    for s in random.sample(problematic, min(len(problematic), max(1, len(problematic) // 4))):
        if random.random() < prob_move:
            tries = []
            for d, ordered in day_ordered.items():
                for i in range(len(ordered)):
                    st = ordered[i]
                    if find_consecutive(st, s.length, id_to_slot, day_ordered):
                        tries.append(st)
            random.shuffle(tries)
            cands = candidates.get((s.id_kelas, s.id_mapel), [])
            if not cands:
                continue
            for st in tries[:60]:
                block = find_consecutive(st, s.length, id_to_slot, day_ordered)
                if not block:
                    continue
                d = id_to_slot[block[0]]["hari"]
                # don't place if same class already has this mapel today
                if per_class_mapel_day[s.nama_kelas].get((d, s.id_mapel), 0) > 0:
                    continue
                # don't place if split_group already on that day
                if s.split_group is not None and (d in placed_split_days[s.split_group]):
                    continue
                for g in random.sample(cands, min(len(cands), 5)):
                    # quick teacher conflict check: avoid immediate conflicts on block
                    conflict = False
                    for wid in block:
                        if any(g == guru_map.get(other_sid) and find_consecutive(guru_map and slots_map.get(other_sid, -1), s.length, id_to_slot, day_ordered) for other_sid in []):
                            pass  # noop - we will let fitness detect complex conflict
                    # accept move
                    slots_map[s.sid] = st
                    guru_map[s.sid] = g
                    # update local trackers
                    per_class_mapel_day[s.nama_kelas][(d, s.id_mapel)] += s.length
                    if s.split_group is not None:
                        placed_split_days[s.split_group].add(d)
                    break
                else:
                    continue
                break

    # swap: swap sessions between two classes for same length
    if random.random() < prob_swap:
        bylen = defaultdict(list)
        for s in sessions:
            bylen[s.length].append(s)
        lengths = [L for L in bylen if len(bylen[L]) >= 2]
        if lengths:
            L = random.choice(lengths)
            s1, s2 = random.sample(bylen[L], 2)
            tmp1 = slots_map.get(s1.sid); tmpg1 = guru_map.get(s1.sid)
            tmp2 = slots_map.get(s2.sid); tmpg2 = guru_map.get(s2.sid)
            # before swapping check day-mapel constraints
            ok = True
            if tmp2:
                block2 = find_consecutive(tmp2, s2.length, id_to_slot, day_ordered)
                if block2:
                    d2 = id_to_slot[block2[0]]["hari"]
                    if per_class_mapel_day[s1.nama_kelas].get((d2, s1.id_mapel), 0) > 0 and (slots_map.get(s1.sid) is None or id_to_slot[slots_map[s1.sid]]["hari"] != d2):
                        ok = False
            if tmp1:
                block1 = find_consecutive(tmp1, s1.length, id_to_slot, day_ordered)
                if block1:
                    d1 = id_to_slot[block1[0]]["hari"]
                    if per_class_mapel_day[s2.nama_kelas].get((d1, s2.id_mapel), 0) > 0 and (slots_map.get(s2.sid) is None or id_to_slot[slots_map[s2.sid]]["hari"] != d1):
                        ok = False
            if ok:
                slots_map[s1.sid], guru_map[s1.sid] = tmp2, tmpg2
                slots_map[s2.sid], guru_map[s2.sid] = tmp1, tmpg1

    return slots_map, guru_map

# ---------------- Repair local (attempt to improve individual before evaluation) ----------------
def repair_local(ind, sessions, id_to_slot, day_ordered, candidates, max_iter=200):
    slots_map, guru_map = deepcopy(ind[0]), deepcopy(ind[1])
    per_class_assigned = defaultdict(set)
    per_class_mapel_day = defaultdict(lambda: defaultdict(int))
    placed_split_days = defaultdict(set)

    # initialize trackers
    for s in sessions:
        st = slots_map.get(s.sid)
        if st is None:
            continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block:
            continue
        cls = s.nama_kelas
        d = id_to_slot[block[0]]["hari"]
        per_class_assigned[cls].update(block)
        per_class_mapel_day[cls][(d, s.id_mapel)] += s.length
        if s.split_group is not None:
            placed_split_days[s.split_group].add(d)

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
            if not block:
                continue
            cls = s.nama_kelas
            d = id_to_slot[block[0]]["hari"]
            if any(w in per_class_assigned[cls] for w in block):
                continue
            if per_class_mapel_day[cls].get((d, s.id_mapel), 0) > 0:
                continue
            if s.split_group is not None and (d in placed_split_days[s.split_group]):
                continue
            for g in random.sample(cands, min(len(cands), 5)):
                slots_map[s.sid] = st
                guru_map[s.sid] = g
                for wid in block:
                    per_class_assigned[cls].add(wid)
                per_class_mapel_day[cls][(d, s.id_mapel)] += s.length
                if s.split_group is not None:
                    placed_split_days[s.split_group].add(d)
                placed = True
                break
            if placed:
                break
        if not placed:
            slots_map[s.sid] = None
            guru_map[s.sid] = None
    return slots_map, guru_map

# ---------------- Save generation/final ----------------
def save_grid(run_dir, gen_idx, chrom, sessions, id_to_slot, ordered_slots, day_ordered, tables, tag="gen"):
    sub = f"{tag}_{gen_idx:03d}" if tag == "gen" else "final"
    dirp = os.path.join(run_dir, sub) if tag == "gen" else os.path.join(run_dir, "final")
    os.makedirs(dirp, exist_ok=True)

    slots_map, guru_map = chrom

    rows = []
    for wid in ordered_slots:
        r = id_to_slot[wid]
        rows.append({"id_waktu": wid, "hari": r["hari"], "jam_ke": r["jam_ke"]})
    df_slots = pd.DataFrame(rows)

    class_names = sorted({s.nama_kelas for s in sessions})
    table = pd.DataFrame(index=range(len(df_slots)), columns=["hari", "jam_ke"] + class_names)
    table["hari"] = df_slots["hari"]
    table["jam_ke"] = df_slots["jam_ke"]
    table.fillna("", inplace=True)
    guru_df = tables["guru"]

    for s in sessions:
        st = slots_map.get(s.sid)
        g = guru_map.get(s.sid)
        if st is None or g is None:
            continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block:
            continue
        try:
            nama_g = guru_df.loc[guru_df["id_guru"] == g, "nama_guru"].iloc[0]
        except:
            nama_g = str(g)
        entry = f"{nama_g} - {s.nama_mapel}"
        for wid in block:
            idx = df_slots.index[df_slots["id_waktu"] == wid].tolist()
            if not idx:
                continue
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

    missing_candidates = []
    for s in sessions:
        if len(candidates.get((s.id_kelas, s.id_mapel), [])) == 0:
            missing_candidates.append((s.nama_kelas, s.nama_mapel))
    if missing_candidates:
        print("[WARN] Some sessions have no candidate teacher (examples class,mapel):")
        print(missing_candidates[:20])
        print("Please add active teachers in guru_mapel for these pairs.")

    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT, f"run_{run_label}")
    os.makedirs(run_dir, exist_ok=True)
    for k, v in tables.items():
        try:
            v.to_csv(os.path.join(run_dir, f"{k}.csv"), index=False)
        except:
            pass

    population = create_initial_population(sessions, id_to_slot, day_ordered, candidates, POP_SIZE)
    population = [repair_local(ind, sessions, id_to_slot, day_ordered, candidates, max_iter=300) for ind in population]

    fitnesses = []
    diags = []
    for ind in population:
        f, d = fitness_score(ind, sessions, id_to_slot, day_ordered, expected_per_class, candidates, guru_capacity)
        fitnesses.append(f)
        diags.append(d)
    print(f"[info] init avg={sum(fitnesses)/len(fitnesses):.6f} best={max(fitnesses):.6f}")

    best_overall = None
    best_f_overall = -1e12

    for gen in range(1, GENERATIONS + 1):
        avg = sum(fitnesses) / len(fitnesses)
        best_f = max(fitnesses)
        worst_f = min(fitnesses)
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        print(f"[Gen {gen:03d}] avg={avg:.6f} best={best_f:.6f} worst={worst_f:.6f}")

        # save best of generation
        save_grid(run_dir, gen, population[best_idx], sessions, id_to_slot, ordered_slots, day_ordered, tables, tag="gen")

        if best_f > best_f_overall:
            best_f_overall = best_f
            best_overall = deepcopy(population[best_idx])

        new_pop = [deepcopy(population[best_idx])]  # elitism
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

        fitnesses = []
        diags = []
        for ind in population:
            f, d = fitness_score(ind, sessions, id_to_slot, day_ordered, expected_per_class, candidates, guru_capacity)
            fitnesses.append(f)
            diags.append(d)

        # early stop heuristic
        potential_perfect = 40 * len(set([s.nama_kelas for s in sessions]))
        if best_f_overall >= potential_perfect:
            print("[INFO] Achieved heuristic perfect score, stopping early.")
            break

    final_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    final_fname, final_table = save_grid(run_dir, 0, population[final_idx], sessions, id_to_slot, ordered_slots, day_ordered, tables, tag="final")
    print(f"[INFO] final saved to {os.path.dirname(final_fname)}")
    print(f"[INFO] best overall score: {best_f_overall:.6f}")
    best_score, best_diag = fitness_score(population[final_idx], sessions, id_to_slot, day_ordered, expected_per_class, candidates, guru_capacity)
    print("[INFO] best final diagnostics sample:", best_diag)
    with pd.option_context("display.max_rows", 200, "display.max_columns", None):
        print(final_table.fillna("").head(40).to_string(index=False))

    ans = input("Simpan final ke database jadwal_master/jadwal? (y/n): ").strip().lower()
    if ans == "y":
        slots_map, guru_map = population[final_idx]
        tahun = input("Tahun ajaran (contoh: 2025/2026): ").strip()
        semester = input("Semester (ganjil/genap): ").strip()
        keterangan = input("Keterangan (opsional): ").strip()
        with engine.begin() as conn:
            res = conn.execute(
                text("INSERT INTO jadwal_master (tahun_ajaran, semester, keterangan, dibuat_pada) VALUES (:t,:s,:k,CURRENT_TIMESTAMP())"),
                {"t": tahun, "s": semester, "k": keterangan},
            )
            master_id = res.lastrowid
            rows = []
            for s in sessions:
                st = slots_map.get(s.sid)
                g = guru_map.get(s.sid)
                if st is None or g is None:
                    continue
                block = find_consecutive(st, s.length, id_to_slot, day_ordered)
                if not block:
                    continue
                for wid in block:
                    rows.append(
                        {
                            "id_master": master_id,
                            "id_kelas": s.id_kelas,
                            "id_mapel": s.id_mapel,
                            "id_guru": g,
                            "id_ruang": None,
                            "id_waktu": wid,
                            "generasi": None,
                            "fitness": None,
                        }
                    )
            if rows:
                df = pd.DataFrame(rows)
                df.to_sql("jadwal", con=conn, if_exists="append", index=False)
        print(f"[DB] saved jadwal_master id {master_id} with {len(rows)} rows.")
    print("Done.")

if __name__ == "__main__":
    run_ga()
