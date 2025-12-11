#!/usr/bin/env python3
"""
ga_scheduler.py
Algoritma Genetika untuk penjadwalan mata pelajaran sesuai constraint user.
Simpan sebagai file .py lalu jalankan.

Penggunaan:
    python ga_scheduler.py --host HOST --user USER --password PASS --db DBNAME
"""

import argparse
import pymysql
import pandas as pd
import numpy as np
import os
import json
import random
from datetime import datetime
from collections import defaultdict, namedtuple, Counter
import math
import shutil
from dotenv import load_dotenv
import os

# --------------------------
# Konfigurasi / Default GA
# --------------------------
DEFAULTS = {
    "pop_size": 120,
    "generations": 100,
    "crossover_prob": 0.8,
    "mutation_prob": 0.12,
    "elitism": 2,
    "tournament_k": 3,
    "seed": 42
}

# --------------------------
# Utilities
# --------------------------
def mkdir_p(path):
    os.makedirs(path, exist_ok=True)

# --------------------------
# Database loader
# --------------------------
def load_db(host, user, password, db, port=3306):
    conn = pymysql.connect(host=host, user=user, password=password, database=db, port=port, charset='utf8mb4')
    return conn

def read_tables(conn):
    # read necessary tables into pandas
    df_guru = pd.read_sql("SELECT * FROM guru", conn)
    df_mapel = pd.read_sql("SELECT * FROM mapel", conn)
    df_kelas = pd.read_sql("SELECT * FROM kelas", conn)
    df_waktu = pd.read_sql("SELECT * FROM waktu ORDER BY FIELD(hari,'Senin','Selasa','Rabu','Kamis','Jumat'), jam_ke", conn)
    df_guru_mapel = pd.read_sql("SELECT * FROM guru_mapel WHERE aktif='aktif'", conn)
    df_konf = pd.read_sql("SELECT * FROM konfigurasi_ag LIMIT 1", conn) if table_exists(conn, "konfigurasi_ag") else pd.DataFrame()
    return df_guru, df_mapel, df_kelas, df_waktu, df_guru_mapel, df_konf

def table_exists(conn, tbl):
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE %s", (tbl,))
        return cur.rowcount > 0

# --------------------------
# Session generation
# --------------------------
Session = namedtuple("Session", ["sid","id_kelas","id_mapel","id_guru","duration","tag","split_group"])
# tag: for readability "Matematika_1" etc
# split_group: None or group id to enforce day-different constraint (for mapel split into 2)

def generate_sessions(df_guru_mapel, df_mapel, df_kelas):
    """
    Build list of sessions from active guru_mapel.
    Rules:
      - For each mapping (guru teaches mapel to kelas), consult mapel.jam_per_minggu.
      - If jam_per_minggu <= 3 => create jam_per_minggu sessions of duration 1 each.
      - If jam_per_minggu > 3 => split into 2 sessions (ceil/ floor), each session duration >1 possible:
          * Example: 5 -> [3,2] (durasi per meeting)
        Additionally: place two split sessions in different days (we will enforce in fitness/repair).
      - Each session is assigned a unique sid.
    """
    sessions = []
    sid = 0
    split_counter = 0
    mapel_jam = df_mapel.set_index('id_mapel')['jam_per_minggu'].to_dict()
    for _, row in df_guru_mapel.iterrows():
        id_guru = int(row['id_guru'])
        id_mapel = int(row['id_mapel'])
        id_kelas = int(row['id_kelas'])
        jam = int(mapel_jam.get(id_mapel, 1))
        nama_mapel = df_mapel.loc[df_mapel['id_mapel']==id_mapel,'nama_mapel'].iloc[0] if id_mapel in df_mapel['id_mapel'].values else f"Mapel{int(id_mapel)}"

        if jam <= 3:
            # create jam sessions of duration 1
            for i in range(jam):
                sid += 1
                sessions.append(Session(sid,id_kelas,id_mapel,id_guru,1,f"{nama_mapel}",None))
        else:
            # split into 2 meetings: a = ceil(jam/2), b = jam - a
            a = math.ceil(jam/2)
            b = jam - a
            split_counter += 1
            # we create one session duration=a, one duration=b
            sid += 1
            sessions.append(Session(sid,id_kelas,id_mapel,id_guru,a,f"{nama_mapel}_splitA",split_counter))
            sid += 1
            sessions.append(Session(sid,id_kelas,id_mapel,id_guru,b,f"{nama_mapel}_splitB",split_counter))
    return sessions

# --------------------------
# Slot building from waktu
# --------------------------
Slot = namedtuple("Slot", ["slot_id","hari","jam_ke","start","end","keterangan","index_in_day"])
def build_slots(df_waktu):
    """
    Build list of available atomic slots.
    We'll only use waktu rows where jam_ke is NOT NULL and keterangan is NULL (per constraint d).
    Also we need to index slots by day and jam_ke to find consecutive sequences for multi-hour sessions.
    """
    slots = []
    index = 0
    # keep only allowed slots
    valid = df_waktu[(df_waktu['jam_ke'].notnull()) & (df_waktu['keterangan'].isnull())].copy()
    # group by hari and order by jam_ke
    grouped = valid.groupby('hari', sort=False)
    slot_id = 0
    for hari, g in grouped:
        g_sorted = g.sort_values('jam_ke')
        # index_in_day to help find consecutive sequences
        for idx, row in g_sorted.iterrows():
            slot_id += 1
            slot = Slot(slot_id, hari, int(row['jam_ke']), str(row['waktu_mulai']), str(row['waktu_selesai']), row['keterangan'], int(row['jam_ke']))
            slots.append(slot)
    return slots

def group_slots_by_day(slots):
    by_day = defaultdict(list)
    for s in slots:
        by_day[s.hari].append(s)
    # ensure sorted by jam_ke
    for hari in by_day:
        by_day[hari].sort(key=lambda x: x.jam_ke)
    return by_day

# find consecutive slot starts per day for a given duration
def possible_start_slots_for_duration(by_day, duration):
    """
    Returns dict: hari -> list of slot_id that can be start (so that duration consecutive slots exist)
    """
    res = {}
    for hari, slots in by_day.items():
        starts = []
        n = len(slots)
        for i in range(n):
            # check if we have duration consecutive jam_ke increasing by 1
            ok = True
            for d in range(duration):
                if i+d >= n or slots[i+d].jam_ke != slots[i].jam_ke + d:
                    ok = False
                    break
            if ok:
                # collect the slot_id of the start
                starts.append(slots[i].slot_id)
        res[hari] = starts
    return res

# mapping slot_id to slot object
def slot_dict(slots):
    return {s.slot_id: s for s in slots}

# --------------------------
# Chromosome representation
# --------------------------
# We'll represent chromosome as dict: session_sid -> assigned_slot_start_id
# For duration>1, assigned_slot_start_id indicates the start slot; the session occupies consecutive slots.

# Helpers to get all occupied slot_ids given a chromosome
def occupied_slots_from_assign(chrom, sessions, slot_map, by_day):
    occ = defaultdict(list)  # session_sid -> list of slot_ids occupied
    slotid_to_session = {}
    for s in sessions:
        sid = s.sid
        start = chrom.get(sid)
        if start is None:
            continue
        # determine day
        slot = slot_map.get(start)
        if slot is None:
            continue
        hari = slot.hari
        # find consecutive based on jam_ke
        # get day slots ordered by jam_ke
        day_slots = by_day[hari]
        # find index of start in day_slots by slot_id
        idx = None
        for i, ds in enumerate(day_slots):
            if ds.slot_id == start:
                idx = i
                break
        if idx is None:
            continue
        # collect slot ids for duration
        ids = []
        for d in range(s.duration):
            if idx + d < len(day_slots):
                ids.append(day_slots[idx + d].slot_id)
        occ[sid] = ids
        for slotid in ids:
            slotid_to_session.setdefault(slotid, []).append(sid)
    return occ, slotid_to_session

# --------------------------
# Fitness
# --------------------------
def evaluate_fitness(chrom, sessions, slot_map, by_day, df_kelas):
    """
    Compute penalty-based fitness. Lower penalty => better. We'll return a numeric fitness where higher is better.
    Penalty components (weights very large for hard constraints):
      - teacher_conflict: teacher assigned to multiple classes at same slot -> penalty_per_conflict * count
      - missing_hours_per_class: if sum(duration) assigned for class != sum(mapel jam) -> penalty_per_hour * diff
      - hole_in_day (gap): if for any class there is a gap in the middle of day (i.e. empty jam between assigned jam on same day) -> penalty big
      - using forbidden time (but we banned them already by building slots) -> large penalty
      - sessions of same split_group in same day -> penalty (we want different days)
      - multi-hour sessions not consecutive (shouldn't happen because we only assign consecutive starts) -> penalty
      - unassigned sessions -> penalty
    We'll compute total_penalty and return fitness = 1e6 - total_penalty so larger is better.
    """
    PEN = {
        "teacher_conflict": 5000,
        "missing_hour": 1000,
        "hole": 3000,
        "split_same_day": 2000,
        "unassigned": 2000,
        "overlap_class": 5000,
        "pref_adjacent": -500  # negative reduces penalty (i.e., reward) if adjacency is satisfied; applied separately
    }

    total_penalty = 0
    sessions_by_sid = {s.sid: s for s in sessions}
    # occupied slots mapping
    occ, slotid_to_session = occupied_slots_from_assign(chrom, sessions, slot_map, by_day)

    # 1) teacher conflicts: for each slotid, check if a teacher appears >1
    teacher_conflicts = 0
    for slotid, sids in slotid_to_session.items():
        if len(sids) <= 1:
            continue
        # check teacher ids
        teachers = [sessions_by_sid[sid].id_guru for sid in sids]
        cnt = len(teachers) - len(set(teachers))
        if cnt > 0:
            teacher_conflicts += cnt
    total_penalty += PEN["teacher_conflict"] * teacher_conflicts

    # 2) class overlap (two sessions same class same time)
    overlap_class = 0
    for slotid, sids in slotid_to_session.items():
        classes = [sessions_by_sid[sid].id_kelas for sid in sids]
        cnt = len(classes) - len(set(classes))
        if cnt > 0:
            overlap_class += cnt
    total_penalty += PEN["overlap_class"] * overlap_class

    # 3) unassigned sessions
    unassigned = sum(1 for s in sessions if s.sid not in chrom or chrom[s.sid] is None)
    total_penalty += PEN["unassigned"] * unassigned

    # 4) missing hours per class vs required
    required_hours = defaultdict(int)
    assigned_hours = defaultdict(int)
    # build required hours from sessions list (sum durations per class)
    for s in sessions:
        required_hours[s.id_kelas] += s.duration
    for s in sessions:
        if s.sid in chrom and chrom[s.sid] is not None:
            assigned_hours[s.id_kelas] += s.duration
    missing = 0
    for k, req in required_hours.items():
        asg = assigned_hours.get(k,0)
        if asg != req:
            diff = abs(req - asg)
            missing += diff
    total_penalty += PEN["missing_hour"] * missing

    # 5) holes (gap) per class per day: we want no gap in middle of day
    # For each class, group assigned slots by hari, then check if assigned jam_ke set is contiguous from min to max
    slot_by_id = {s.slot_id: s for day in by_day.values() for s in day}
    holes = 0
    for kelas in set([s.id_kelas for s in sessions]):
        # mapping hari -> list of jam_ke assigned for this class
        day_jamkes = defaultdict(list)
        for s in sessions:
            if s.id_kelas != kelas:
                continue
            if s.sid in chrom and chrom[s.sid] is not None:
                start = chrom[s.sid]
                slot = slot_map.get(start)
                if slot is None:
                    continue
                hari = slot.hari
                # compute jam_ke list
                # locate day slots
                day_slots = by_day[hari]
                # find start index
                idx = None
                for i, ds in enumerate(day_slots):
                    if ds.slot_id == start:
                        idx = i
                        break
                if idx is None:
                    continue
                for d in range(s.duration):
                    if idx + d < len(day_slots):
                        day_jamkes[hari].append(day_slots[idx+d].jam_ke)
        # evaluate per day
        for hari, jamkes in day_jamkes.items():
            if not jamkes:
                continue
            jamkes_sorted = sorted(set(jamkes))
            if max(jamkes_sorted) - min(jamkes_sorted) + 1 != len(jamkes_sorted):
                holes += 1
    total_penalty += PEN["hole"] * holes

    # 6) split_group same day penalty: for sessions with split_group not None, ensure they are on different hari
    split_same = 0
    groups = defaultdict(list)
    for s in sessions:
        if s.split_group is not None:
            groups[s.split_group].append(s.sid)
    for grp, sids in groups.items():
        days = []
        for sid in sids:
            if sid in chrom and chrom[sid] is not None:
                slot = slot_map.get(chrom[sid])
                if slot:
                    days.append(slot.hari)
        if len(days) >= 2:
            if len(days) != len(set(days)):
                split_same += 1
    total_penalty += PEN["split_same_day"] * split_same

    # 7) reward adjacency for multi-hour (if the assigned slots are consecutive) -> reduce penalty (negative)
    adjacency_reward_count = 0
    for s in sessions:
        if s.duration > 1 and s.sid in chrom and chrom[s.sid] is not None:
            start = chrom[s.sid]
            slot = slot_map.get(start)
            if not slot:
                continue
            hari = slot.hari
            day_slots = by_day[hari]
            idx = None
            for i, ds in enumerate(day_slots):
                if ds.slot_id == start:
                    idx = i
                    break
            ok = True
            if idx is None: ok = False
            else:
                for d in range(s.duration):
                    if idx + d >= len(day_slots) or day_slots[idx+d].jam_ke != day_slots[idx].jam_ke + d:
                        ok = False
                        break
            if ok:
                adjacency_reward_count += 1
    total_penalty += PEN["pref_adjacent"] * adjacency_reward_count  # negative lowers penalty

    # Build fitness: higher is better
    FITNESS_BASE = 10**7
    fitness = FITNESS_BASE - total_penalty
    # Also return breakdown for diagnostics
    diagnostics = {
        "penalty": total_penalty,
        "teacher_conflicts": teacher_conflicts,
        "overlap_class": overlap_class,
        "unassigned": unassigned,
        "missing_hours_total": missing,
        "holes": holes,
        "split_same_day": split_same,
        "adjacency_rewards": adjacency_reward_count
    }
    return fitness, diagnostics

# --------------------------
# Initialization (smart)
# --------------------------
def init_population(pop_size, sessions, by_day, slot_map, slot_starts_by_day):
    """
    Initialize population by assigning sessions randomly but respecting:
      - For duration>1, pick start slot that has enough consecutive slots.
      - Try to avoid immediate teacher conflicts by checking teacher occupancy heuristics.
    """
    population = []
    random.seed(DEFAULTS['seed'])
    session_list = sessions.copy()
    for _ in range(pop_size):
        chrom = {}
        # occupancy per slot id (for this chromosome)
        occ_slots = set()
        # simple heuristic: shuffle sessions then place ones with longer duration first
        random.shuffle(session_list)
        session_list_sorted = sorted(session_list, key=lambda s: -s.duration)
        for s in session_list_sorted:
            # try to find a start slot
            possible_starts = []
            # build combined list of all starts across days
            for hari, starts in slot_starts_by_day.items():
                possible_starts.extend(starts)
            if not possible_starts:
                chrom[s.sid] = None
                continue
            # randomize order
            random.shuffle(possible_starts)
            placed = False
            for start in possible_starts:
                start_slot = slot_map[start]
                hari = start_slot.hari
                day_slots = by_day[hari]
                # locate index
                idx = None
                for i, ds in enumerate(day_slots):
                    if ds.slot_id == start:
                        idx = i
                        break
                if idx is None:
                    continue
                # ensure consecutive exist
                ok = True
                ids = []
                for d in range(s.duration):
                    if idx + d >= len(day_slots):
                        ok = False
                        break
                    ids.append(day_slots[idx+d].slot_id)
                if not ok:
                    continue
                # check if any ids already used (simple placement)
                conflict = any(slotid in occ_slots for slotid in ids)
                if not conflict:
                    # place it
                    chrom[s.sid] = start
                    occ_slots.update(ids)
                    placed = True
                    break
            if not placed:
                chrom[s.sid] = None
        population.append(chrom)
    return population

# --------------------------
# Genetic operators
# --------------------------
def tournament_selection(population, fitnesses, k=3):
    i = random.sample(range(len(population)), k)
    best = max(i, key=lambda idx: fitnesses[idx])
    return population[best].copy()

def crossover(parent1, parent2, sessions, crossover_prob=0.8):
    # subset crossover: choose random subset of sessions to swap assignments
    if random.random() > crossover_prob:
        return parent1.copy(), parent2.copy()
    keys = [s.sid for s in sessions]
    cut = random.randint(1, max(1, len(keys)//3))
    subset = set(random.sample(keys, cut))
    child1 = parent1.copy()
    child2 = parent2.copy()
    for k in subset:
        child1[k], child2[k] = child2.get(k), child1.get(k)
    return child1, child2

def mutate(chrom, sessions, by_day, slot_map, slot_starts_by_day, mutation_prob=0.12):
    # For each session, with prob mutation_prob, move to a different random valid start
    keys = [s.sid for s in sessions]
    for sid in keys:
        if random.random() < mutation_prob:
            s = next(filter(lambda x: x.sid==sid, sessions))
            # choose new start from all possible starts for s.duration
            all_starts = []
            for hari, starts in slot_starts_by_day.items():
                all_starts.extend(starts)
            if not all_starts:
                chrom[sid] = None
                continue
            # pick random that fits consecutive
            random.shuffle(all_starts)
            placed = False
            for start in all_starts:
                slot = slot_map[start]
                hari = slot.hari
                day_slots = by_day[hari]
                idx = None
                for i, ds in enumerate(day_slots):
                    if ds.slot_id == start:
                        idx = i
                        break
                if idx is None:
                    continue
                ok = True
                for d in range(s.duration):
                    if idx + d >= len(day_slots) or day_slots[idx+d].jam_ke != day_slots[idx].jam_ke + d:
                        ok = False
                        break
                if ok:
                    chrom[sid] = start
                    placed = True
                    break
            if not placed:
                chrom[sid] = None
    return chrom

# --------------------------
# Repair / Local Search
# --------------------------
def repair(chrom, sessions, slot_map, by_day):
    """
    Simple repair:
      - Resolve teacher conflicts by moving the later assigned session to another available start.
      - Resolve class overlap similarly.
      - Try to fill unassigned sessions by random valid placement.
    """
    # Build reverse mapping slotid -> sids
    occ, slotid_to_sess = occupied_slots_from_assign(chrom, sessions, slot_map, by_day)
    # teacher conflicts resolution
    sessions_by_sid = {s.sid: s for s in sessions}
    # Try to iterate through slotid_to_sess conflicts
    for slotid, sids in list(slotid_to_sess.items()):
        # if multiple teachers in same slot (teachers duplicated) or same teacher multiple classes
        if len(sids) <= 1:
            continue
        # group by teacher
        by_teacher = defaultdict(list)
        for sid in sids:
            by_teacher[sessions_by_sid[sid].id_guru].append(sid)
        # if any teacher has more than 1 session here, move the excess sessions
        for teacher, teacher_sids in by_teacher.items():
            if len(teacher_sids) <= 1:
                continue
            # keep first, move others
            for sid_to_move in teacher_sids[1:]:
                # find alternative start
                s = sessions_by_sid[sid_to_move]
                moved = False
                # find any start in same day or other day where unoccupied
                for hari, day_slots in by_day.items():
                    for i in range(len(day_slots)):
                        # check consecutive
                        ok = True
                        ids = []
                        for d in range(s.duration):
                            if i + d >= len(day_slots) or day_slots[i+d].jam_ke != day_slots[i].jam_ke + d:
                                ok = False
                                break
                            ids.append(day_slots[i+d].slot_id)
                        if not ok:
                            continue
                        # check current occupancy
                        conflict = False
                        for idd in ids:
                            # if any existing sessions other than this one occupy it -> conflict
                            occ_sess = slotid_to_sess.get(idd, [])
                            # allow if occ_sess is exactly [sid_to_move]
                            if any((x != sid_to_move) for x in occ_sess):
                                conflict = True
                                break
                        if not conflict:
                            # assign it
                            chrom[sid_to_move] = day_slots[i].slot_id
                            moved = True
                            break
                    if moved:
                        break
                if not moved:
                    chrom[sid_to_move] = None
    # try fill unassigned
    for s in sessions:
        if s.sid not in chrom or chrom[s.sid] is None:
            placed = False
            for hari, day_slots in by_day.items():
                for i in range(len(day_slots)):
                    ok = True
                    for d in range(s.duration):
                        if i + d >= len(day_slots) or day_slots[i+d].jam_ke != day_slots[i].jam_ke + d:
                            ok = False
                            break
                    if not ok: continue
                    # ensure no overlap by other assigned sessions
                    ids = [day_slots[i+d].slot_id for d in range(s.duration)]
                    occ2, s2 = occupied_slots_from_assign(chrom, sessions, slot_map, by_day)
                    conflict = False
                    for idd in ids:
                        if idd in s2 and len(s2[idd])>0:
                            conflict = True
                            break
                    if not conflict:
                        chrom[s.sid] = day_slots[i].slot_id
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                chrom[s.sid] = None
    return chrom

# --------------------------
# Save functions
# --------------------------
def save_generation_result(run_dir, gen_idx, chrom, sessions, slot_map, by_day, fitness, diagnostics):
    gen_dir = os.path.join(run_dir, f"gen_{gen_idx:04d}")
    mkdir_p(gen_dir)
    # Save JSON of assignments
    out_assign = []
    occ, slotid_to_session = occupied_slots_from_assign(chrom, sessions, slot_map, by_day)
    for s in sessions:
        start = chrom.get(s.sid)
        if start is None:
            out_assign.append({
                "sid": s.sid, "kelas": s.id_kelas, "mapel": s.id_mapel, "guru": s.id_guru,
                "duration": s.duration, "start_slot": None
            })
        else:
            slot = slot_map[start]
            out_assign.append({
                "sid": s.sid, "kelas": s.id_kelas, "mapel": s.id_mapel, "guru": s.id_guru,
                "duration": s.duration, "start_slot": start, "hari": slot.hari, "jam_ke": slot.jam_ke
            })
    with open(os.path.join(gen_dir, "assignments.json"), "w", encoding="utf-8") as f:
        json.dump({"fitness": fitness, "diagnostics": diagnostics, "assignments": out_assign}, f, indent=2)
    # Save CSV readable schedule per class
    rows = []
    for s in sessions:
        if s.sid in chrom and chrom[s.sid] is not None:
            start = chrom[s.sid]
            slot = slot_map[start]
            hari = slot.hari
            # find jam_ke sequence
            day_slots = by_day[hari]
            idx = None
            for i, ds in enumerate(day_slots):
                if ds.slot_id == start:
                    idx = i
                    break
            jamkes = []
            for d in range(s.duration):
                if idx + d < len(day_slots):
                    jamkes.append(day_slots[idx+d].jam_ke)
            rows.append({"kelas": s.id_kelas, "mapel": s.id_mapel, "guru": s.id_guru, "hari": hari, "jam_ke": ",".join(map(str,jamkes))})
        else:
            rows.append({"kelas": s.id_kelas, "mapel": s.id_mapel, "guru": s.id_guru, "hari": None, "jam_ke": None})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(gen_dir, "schedule.csv"), index=False)
    # also save summary
    with open(os.path.join(gen_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"fitness: {fitness}\n")
        f.write(json.dumps(diagnostics, indent=2))

# --------------------------
# Main GA loop
# --------------------------
def run_ga(conn, params):
    # load tables
    df_guru, df_mapel, df_kelas, df_waktu, df_guru_mapel, df_konf = read_tables(conn)
    sessions = generate_sessions(df_guru_mapel, df_mapel, df_kelas)
    slots = build_slots(df_waktu)
    by_day = group_slots_by_day(slots)
    slot_map = slot_dict(slots)
    slot_starts_by_day = possible_start_slots_for_duration(by_day, 1)  # starts for duration=1 (we'll reuse)
    # but for multi-duration we need starts per duration per day => compute later on demand

    # create results folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"run_{timestamp}")
    mkdir_p(run_dir)

    # Setup GA parameters (allow override from konfigurasi_ag table)
    pop_size = params.get("pop_size", DEFAULTS['pop_size'])
    generations = params.get("generations", DEFAULTS['generations'])
    crossover_prob = params.get("crossover_prob", DEFAULTS['crossover_prob'])
    mutation_prob = params.get("mutation_prob", DEFAULTS['mutation_prob'])
    elitism = params.get("elitism", DEFAULTS['elitism'])
    tournament_k = params.get("tournament_k", DEFAULTS['tournament_k'])
    random.seed(params.get('seed', DEFAULTS['seed']))

    # Prepare slot_starts_by_day for each duration (to speed lookups)
    durations = sorted(list(set([s.duration for s in sessions])))
    slot_starts_by_duration = {}
    for dur in durations:
        slot_starts_by_duration[dur] = possible_start_slots_for_duration(by_day, dur)

    # init population
    population = init_population(pop_size, sessions, by_day, slot_map, slot_starts_by_duration[1])
    # evaluate
    fitnesses = []
    diagnostics_list = []
    for chrom in population:
        fit, diag = evaluate_fitness(chrom, sessions, slot_map, by_day, df_kelas)
        fitnesses.append(fit)
        diagnostics_list.append(diag)
    best_idx = int(np.argmax(fitnesses))
    best_chrom = population[best_idx].copy()
    best_fitness = fitnesses[best_idx]
    print(f"Init: best fitness = {best_fitness} (run dir: {run_dir})")
    save_generation_result(run_dir, 0, best_chrom, sessions, slot_map, by_day, best_fitness, diagnostics_list[best_idx])

    # GA loop
    for gen in range(1, generations+1):
        new_pop = []
        # elitism: keep top N
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        for i in range(elitism):
            new_pop.append(population[ranked[i]].copy())

        # generate rest
        while len(new_pop) < pop_size:
            parent1 = tournament_selection(population, fitnesses, k=tournament_k)
            parent2 = tournament_selection(population, fitnesses, k=tournament_k)
            child1, child2 = crossover(parent1, parent2, sessions, crossover_prob)
            child1 = mutate(child1, sessions, by_day, slot_map, slot_starts_by_duration[1], mutation_prob)
            child2 = mutate(child2, sessions, by_day, slot_map, slot_starts_by_duration[1], mutation_prob)
            # repair
            child1 = repair(child1, sessions, slot_map, by_day)
            child2 = repair(child2, sessions, slot_map, by_day)
            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)

        # evaluate new population
        population = new_pop
        fitnesses = []
        diagnostics_list = []
        for chrom in population:
            fit, diag = evaluate_fitness(chrom, sessions, slot_map, by_day, df_kelas)
            fitnesses.append(fit)
            diagnostics_list.append(diag)
        # update best
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = fitnesses[gen_best_idx]
        gen_best_chrom = population[gen_best_idx].copy()
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_chrom = gen_best_chrom.copy()

        # print status
        avg_fit = sum(fitnesses)/len(fitnesses)
        worst_fit = min(fitnesses)
        print(f"[Gen {gen:04d}] best={gen_best_fit} avg={avg_fit:.2f} worst={worst_fit}")

        # save generation best to results
        save_generation_result(run_dir, gen, gen_best_chrom, sessions, slot_map, by_day, gen_best_fit, diagnostics_list[gen_best_idx])

    # end GA
    print("GA selesai.")
    print(f"Best fitness overall: {best_fitness}")
    # save final best
    save_generation_result(run_dir, "best", best_chrom, sessions, slot_map, by_day, best_fitness, {"note":"final_best"})
    # also print readable schedule for best
    print("Jadwal generasi terbaik (ringkasan):")
    # produce per-class schedule
    occ, slotid_to_session = occupied_slots_from_assign(best_chrom, sessions, slot_map, by_day)
    rows = []
    for s in sessions:
        if s.sid in best_chrom and best_chrom[s.sid] is not None:
            start = best_chrom[s.sid]
            slot = slot_map[start]
            hari = slot.hari
            # compute jamkes
            day_slots = by_day[hari]
            idx = None
            for i, ds in enumerate(day_slots):
                if ds.slot_id == start:
                    idx = i
                    break
            jamkes = []
            for d in range(s.duration):
                if idx + d < len(day_slots):
                    jamkes.append(day_slots[idx+d].jam_ke)
            rows.append((s.id_kelas, s.id_mapel, s.id_guru, hari, jamkes))
        else:
            rows.append((s.id_kelas, s.id_mapel, s.id_guru, None, None))
    # print compact
    df_print = pd.DataFrame(rows, columns=["kelas","mapel","guru","hari","jam_ke"])
    print(df_print.head(60).to_string(index=False))

    # Ask user to save to DB
    ans = input("Simpan jadwal terbaik ke database? (y/n): ").strip().lower()
    if ans == 'y':
        save_to_db(conn, best_chrom, sessions, slot_map, by_day)
        print("Jadwal tersimpan ke tabel jadwal_master dan jadwal.")
    else:
        print("Tidak disimpan.")

    print(f"Hasil run disimpan di folder: {run_dir}")
    return run_dir

# --------------------------
# Save to DB
# --------------------------
def save_to_db(conn, chrom, sessions, slot_map, by_day):
    # create jadwal_master entry
    tahun_ajaran = input("Masukkan tahun ajaran (mis: 2025/2026): ").strip()
    semester = input("Masukkan semester (ganjil/genap): ").strip()
    keterangan = input("Keterangan (opsional): ").strip()
    with conn.cursor() as cur:
        cur.execute("INSERT INTO jadwal_master (tahun_ajaran, semester, keterangan) VALUES (%s,%s,%s)", (tahun_ajaran, semester, keterangan))
        master_id = cur.lastrowid
        # insert each assigned session as jadwal row
        for s in sessions:
            if s.sid in chrom and chrom[s.sid] is not None:
                start = chrom[s.sid]
                # we store id_waktu as the start slot's id_waktu - need mapping from slot_id -> original id_waktu
                # but our Slot namedtuple currently doesn't have original id_waktu; we used slot_id synthetic.
                # We'll store id_waktu as None (or extend slot building to carry original id_waktu if needed).
                # For portability, store id_waktu as start (synthetic). If you prefer original id_waktu,
                # modify build_slots to include original id_waktu.
                id_waktu = start
                cur.execute("INSERT INTO jadwal (id_master, id_kelas, id_mapel, id_guru, id_waktu) VALUES (%s,%s,%s,%s,%s)",
                            (master_id, s.id_kelas, s.id_mapel, s.id_guru, id_waktu))
        conn.commit()
    return True

# --------------------------
# CLI and run
# --------------------------
def main():
    # load .env
    load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default=os.getenv("HOST"))
    parser.add_argument("--user", default=os.getenv("USER"))
    parser.add_argument("--password", default=os.getenv("PASSWORD", ""))
    parser.add_argument("--db", default=os.getenv("DB"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 3306)))

    parser.add_argument("--pop", type=int, default=DEFAULTS['pop_size'])
    parser.add_argument("--gens", type=int, default=DEFAULTS['generations'])
    parser.add_argument("--cross", type=float, default=DEFAULTS['crossover_prob'])
    parser.add_argument("--mut", type=float, default=DEFAULTS['mutation_prob'])
    parser.add_argument("--seed", type=int, default=DEFAULTS['seed'])

    args = parser.parse_args()

    conn = load_db(args.host, args.user, args.password, args.db, port=args.port)
    params = {
        "pop_size": args.pop,
        "generations": args.gens,
        "crossover_prob": args.cross,
        "mutation_prob": args.mut,
        "seed": args.seed
    }
    run_ga(conn, params)
    conn.close()

if __name__ == "__main__":
    main()
