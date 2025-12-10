#!/usr/bin/env python3
"""
ga_global_hard_final.py
Global GA — HARD constraint, session-aware, final version.

Output:
 results/run_<timestamp>/gen_001/combined_timetable.csv
 ...
 results/run_<timestamp>/final/combined_timetable.csv

Format combined CSV = grid (hari,jam_ke,CLASS1,CLASS2,...)
Cell = "NamaGuru - NamaMapel"

Author: Assistant (adapted to user's DB)
"""

import os, random, json, time
from datetime import datetime
from collections import defaultdict, namedtuple
from copy import deepcopy
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import trange

load_dotenv()

# ========== CONFIG ==========
DB_USER = os.getenv("DB_USER","root")
DB_PASS = os.getenv("DB_PASS","")
DB_HOST = os.getenv("DB_HOST","127.0.0.1")
DB_PORT = os.getenv("DB_PORT","3306")
DB_NAME = os.getenv("DB_NAME","db_penjadwalan")

RESULTS_ROOT = os.getenv("RESULTS_DIR","./results")
POP_SIZE = int(os.getenv("POP_SIZE","20"))
GENERATIONS = int(os.getenv("GENERATIONS","80"))

CONN_STR = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONN_STR, pool_pre_ping=True)

random.seed(42)

# ========== TYPES ==========
Session = namedtuple("Session", ["sid","id_kelas","nama_kelas","id_mapel","nama_mapel","length","split_group"])

# ========== DB read ==========
def read_tables():
    with engine.connect() as conn:
        tables = {}
        for t in ["guru","mapel","kelas","guru_mapel","waktu"]:
            tables[t] = pd.read_sql(text(f"SELECT * FROM {t}"), conn)
    return tables

# ========== Slots from DB ==========
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

# ========== Build sessions (global) ==========
def build_sessions(tables, target_hours_per_class=40):
    gm = tables["guru_mapel"][tables["guru_mapel"]["aktif"]=="aktif"]
    mapel_df = tables["mapel"]
    kelas_df = tables["kelas"]
    sessions = []
    sid = 1
    split_gid = 1
    expected = defaultdict(int)
    pairs = gm.groupby(["id_kelas","id_mapel"]).size().reset_index()[["id_kelas","id_mapel"]]
    for _, r in pairs.iterrows():
        id_kelas = int(r["id_kelas"])
        id_mapel = int(r["id_mapel"])
        kelas_row = kelas_df[kelas_df["id_kelas"]==id_kelas].iloc[0]
        mapel_row = mapel_df[mapel_df["id_mapel"]==id_mapel].iloc[0]
        nama_kelas = kelas_row["nama_kelas"]
        nama_mapel = mapel_row["nama_mapel"]
        jam = int(mapel_row["jam_per_minggu"])
        # mapping rules provided
        if jam == 1:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,1,None)); sid+=1
        elif jam == 2:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,2,None)); sid+=1
        elif jam == 3:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,3,None)); sid+=1
        elif jam == 4:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,2,split_gid)); sid+=1
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,2,split_gid)); sid+=1
            split_gid+=1
        elif jam == 5:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,3,split_gid)); sid+=1
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,2,split_gid)); sid+=1
            split_gid+=1
        elif jam == 6:
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,3,split_gid)); sid+=1
            sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,3,split_gid)); sid+=1
            split_gid+=1
        else:
            # fallback: break into 1-hour sessions
            for _ in range(jam):
                sessions.append(Session(sid,id_kelas,nama_kelas,id_mapel,nama_mapel,1,None)); sid+=1
        expected[nama_kelas] += jam
    # override expected to fixed target (user confirmed 40)
    for k in expected:
        expected[k] = target_hours_per_class
    return sessions, expected

# ========== Candidate map (guru options) ==========
def build_candidates(guru_mapel_df):
    cand = defaultdict(list)
    for _, r in guru_mapel_df[guru_mapel_df["aktif"]=="aktif"].iterrows():
        cand[(int(r["id_kelas"]), int(r["id_mapel"]))].append(int(r["id_guru"]))
    return cand

# ========== Utilities: find consecutive block ==========
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
    prev=None
    for wid in block:
        jk = id_to_slot[wid]["jam_ke"]
        if prev is not None and jk != prev+1:
            return None
        prev=jk
    return block

# ========== Initial population (attempt full placement) ==========
def greedy_place_all(sessions, id_to_slot, day_ordered, candidates, ordered_class_names):
    """
    Try greedy backtracking to place all sessions class-by-class.
    Return slots_map, guru_map. If cannot place some session, leave None.
    Strategy:
      - For each class (order provided), place longer sessions first.
      - For each session, try days in natural order and possible starts.
      - Choose first tutor available (randomized) that doesn't conflict with current busy_map.
    """
    slots_map = {}
    guru_map = {}
    busy = {}  # (guru,wid) -> True
    class_assigned = defaultdict(set)
    # group sessions by class
    per_class = defaultdict(list)
    for s in sessions:
        per_class[s.nama_kelas].append(s)
    for cls in ordered_class_names:
        # process sessions of this class, longest first
        if cls not in per_class: continue
        slist = sorted(per_class[cls], key=lambda x: -x.length)
        for s in slist:
            placed=False
            days = ["Senin","Selasa","Rabu","Kamis","Jumat"]
            for d in days:
                ordered = day_ordered.get(d,[])
                for i in range(len(ordered)):
                    start = ordered[i]
                    block = find_consecutive(start, s.length, id_to_slot, day_ordered)
                    if not block: continue
                    # avoid conflict with this class existing assigned slots (no overlap)
                    if any(w in class_assigned[cls] for w in block): continue
                    # try candidate teachers
                    cands = candidates.get((s.id_kelas, s.id_mapel), [])
                    random.shuffle(cands)
                    for g in cands:
                        conflict=False
                        for wid in block:
                            if busy.get((g,wid), False):
                                conflict=True; break
                        if conflict: continue
                        # assign
                        slots_map[s.sid] = block[0]
                        guru_map[s.sid] = g
                        for wid in block:
                            busy[(g,wid)] = True
                            class_assigned[cls].add(wid)
                        placed=True
                        break
                    if placed: break
                if placed: break
            if not placed:
                slots_map[s.sid]=None
                guru_map[s.sid]=None
    return slots_map, guru_map

# ========== Create initial population ==========
def create_initial_population(sessions, id_to_slot, day_ordered, candidates, pop_size):
    class_names = sorted({s.nama_kelas for s in sessions})
    pop=[]
    for i in range(pop_size):
        # vary class order per individual to add diversity
        order = class_names.copy()
        random.shuffle(order)
        slots_map, guru_map = greedy_place_all(sessions, id_to_slot, day_ordered, candidates, order)
        pop.append((slots_map, guru_map))
    return pop

# ========== Fitness (HARD) ==========
def fitness_hard(chrom, sessions, id_to_slot, day_ordered, ordered_slots, expected_per_class, candidates, guru_capacity):
    slots_map, guru_map = chrom
    # critical violations -> immediate invalid
    # violation if: any session not assigned, any block non-consecutive, any teacher/class clash, any slot on keterangan
    teacher_time = defaultdict(list)  # (g,wid)->list
    class_time = defaultdict(list)    # (cls,wid)->list
    assigned_per_class = defaultdict(set)
    split_days = defaultdict(set)
    # validate
    for s in sessions:
        start = slots_map.get(s.sid)
        g = guru_map.get(s.sid)
        if start is None or g is None:
            return 0.0, {"reason":"missing_session","sid":s.sid}
        block = find_consecutive(start, s.length, id_to_slot, day_ordered)
        if not block:
            return 0.0, {"reason":"non_consecutive","sid":s.sid}
        # check keterangan safety (slot table filtered earlier but double-check)
        for wid in block:
            row = id_to_slot.get(wid)
            if row is None:
                return 0.0, {"reason":"invalid_slot","sid":s.sid}
            if row["keterangan"] is not None and str(row["keterangan"]).strip()!="":
                return 0.0, {"reason":"blocked_slot","sid":s.sid,"wid":wid}
        # register
        for wid in block:
            teacher_time[(g,wid)].append(s.sid)
            class_time[(s.nama_kelas,wid)].append(s.sid)
            assigned_per_class[s.nama_kelas].add(wid)
        if s.split_group is not None:
            split_days[s.split_group].add(id_to_slot[block[0]]["hari"])
    # check teacher clash
    for (g,w), lst in teacher_time.items():
        if len(lst) > 1:
            return 0.0, {"reason":"teacher_conflict","guru":g,"wid":w,"count":len(lst)}
    # check class clash
    for (cls,w), lst in class_time.items():
        if len(lst) > 1:
            return 0.0, {"reason":"class_conflict","kelas":cls,"wid":w,"count":len(lst)}
    # split groups must be on different days
    for gid, days in split_days.items():
        if len(days) < 2:
            return 0.0, {"reason":"split_violation","split_group":gid}
    # no gaps per class per day
    for cls, wids in assigned_per_class.items():
        per_day = defaultdict(list)
        for wid in wids:
            per_day[id_to_slot[wid]["hari"]].append(id_to_slot[wid]["jam_ke"])
        for day, jks in per_day.items():
            jks_sorted = sorted(jks)
            for i in range(len(jks_sorted)-1):
                if jks_sorted[i+1] != jks_sorted[i] + 1:
                    return 0.0, {"reason":"gap","kelas":cls,"day":day}
    # expected hours per class
    for cls, expected in expected_per_class.items():
        assigned = len(assigned_per_class.get(cls,set()))
        if assigned != expected:
            return 0.0, {"reason":"expected_hours_mismatch","kelas":cls,"assigned":assigned,"expected":expected}
    # teacher capacity soft check (if any exceed, return 0 too in hard mode)
    teacher_hours = defaultdict(int)
    for (g,w), lst in teacher_time.items():
        teacher_hours[g] += len(lst)
    for g,h in teacher_hours.items():
        cap = guru_capacity.get(g, 9999)
        if h > cap:
            return 0.0, {"reason":"teacher_overload","guru":g,"hours":h,"cap":cap}
    # if passed all hard checks, compute high fitness based on distribution quality (soft)
    # soft metrics: balanced days, less same teacher consecutive overload etc.
    # For simplicity, compute small scoring: base 1.0 and add small bonus for balanced distribution
    # We'll compute variance of hours per class per day (lower better)
    bonus = 0.0
    # compute per-class day counts
    for cls in assigned_per_class:
        per_day = defaultdict(int)
        for wid in assigned_per_class[cls]:
            per_day[id_to_slot[wid]["hari"]] += 1
        vals = list(per_day.values())
        if len(vals)>0:
            var = (sum((v - (sum(vals)/len(vals)))**2 for v in vals)/len(vals))
            bonus += max(0, 1.0 - var/10.0)  # small positive
    fitness = 1.0 + bonus
    return fitness, {"reason":"valid","bonus":bonus}

# ========== GA Operators (chunk-safe) ==========
def tournament(pop, fitnesses, k=3):
    idxs = random.sample(range(len(pop)), min(k, len(pop)))
    best = max(idxs, key=lambda i: fitnesses[i])
    return deepcopy(pop[best])

def chunk_safe_crossover(p1, p2, sessions, prob=0.85):
    if random.random() > prob:
        return deepcopy(p1), deepcopy(p2)
    # uniform swap per session (entire session slot+guru)
    a_slots, a_gurus = deepcopy(p1[0]), deepcopy(p1[1])
    b_slots, b_gurus = deepcopy(p2[0]), deepcopy(p2[1])
    for s in sessions:
        if random.random() < 0.5:
            a_slots[s.sid], b_slots[s.sid] = b_slots.get(s.sid), a_slots.get(s.sid)
            a_gurus[s.sid], b_gurus[s.sid] = b_gurus.get(s.sid), a_gurus.get(s.sid)
    return (a_slots,a_gurus),(b_slots,b_gurus)

def chunk_safe_mutation(ind, sessions, id_to_slot, day_ordered, candidates, prob=0.12):
    slots_map, guru_map = deepcopy(ind[0]), deepcopy(ind[1])
    # mutate by reassigning session to a different valid start &/or guru
    for s in sessions:
        if random.random() < prob:
            # pick random valid start
            starts=[]
            for d, ordered in day_ordered.items():
                for i in range(len(ordered)):
                    st = ordered[i]
                    block = find_consecutive(st, s.length, id_to_slot, day_ordered)
                    if block: starts.append(st)
            if starts:
                slots_map[s.sid] = random.choice(starts)
        if random.random() < prob:
            cands = candidates.get((s.id_kelas, s.id_mapel), [])
            if cands:
                guru_map[s.sid] = random.choice(cands)
    return (slots_map, guru_map)

# ========== Repair (lightweight, keep chunks) ==========
def repair_light(ind, sessions, id_to_slot, day_ordered, candidates):
    slots_map, guru_map = deepcopy(ind[0]), deepcopy(ind[1])
    # mark used blocks per class to avoid class overlapping
    class_assigned = defaultdict(set)
    for s in sessions:
        st = slots_map.get(s.sid)
        g = guru_map.get(s.sid)
        if st is None or g is None:
            slots_map[s.sid]=None; guru_map[s.sid]=None
            continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block:
            slots_map[s.sid]=None; guru_map[s.sid]=None
            continue
        for wid in block:
            class_assigned[s.nama_kelas].add(wid)
    # fill unassigned greedily
    for s in sessions:
        if slots_map.get(s.sid) is not None and guru_map.get(s.sid) is not None:
            continue
        placed=False
        for d in ["Senin","Selasa","Rabu","Kamis","Jumat"]:
            ordered = day_ordered.get(d,[])
            for i in range(len(ordered)):
                st = ordered[i]
                block = find_consecutive(st, s.length, id_to_slot, day_ordered)
                if not block: continue
                if any(w in class_assigned[s.nama_kelas] for w in block): continue
                cands = candidates.get((s.id_kelas, s.id_mapel), [])
                for g in cands:
                    # accept (ignore teacher clash here; fitness will eliminate)
                    slots_map[s.sid]=block[0]; guru_map[s.sid]=g
                    for w in block: class_assigned[s.nama_kelas].add(w)
                    placed=True; break
                if placed: break
            if placed: break
    return (slots_map, guru_map)

# ========== Save generation and final ==========
def save_generation(run_dir, gen_idx, population, fitnesses, best_idx, sessions, id_to_slot, ordered_slots, tables):
    gen_dir = os.path.join(run_dir, f"gen_{gen_idx:03d}")
    os.makedirs(gen_dir, exist_ok=True)
    best = population[best_idx]
    best_f = fitnesses[best_idx]
    # build combined grid
    rows=[]
    for wid in ordered_slots:
        r = id_to_slot[wid]
        rows.append({"id_waktu":wid,"hari":r["hari"],"jam_ke":r["jam_ke"]})
    df_slots = pd.DataFrame(rows)
    class_names = sorted({s.nama_kelas for s in sessions})
    table = pd.DataFrame(index=range(len(df_slots)), columns=["hari","jam_ke"]+class_names)
    table["hari"]=df_slots["hari"]; table["jam_ke"]=df_slots["jam_ke"]
    table.fillna("", inplace=True)
    guru_df = tables["guru"]
    slots_map, guru_map = best
    for s in sessions:
        st = slots_map.get(s.sid)
        g = guru_map.get(s.sid)
        if st is None or g is None: continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block: continue
        try:
            nama_g = guru_df.loc[guru_df["id_guru"]==g,"nama_guru"].iloc[0]
        except:
            nama_g = str(g)
        entry = f"{nama_g} - {s.nama_mapel}"
        for wid in block:
            idx = df_slots.index[df_slots["id_waktu"]==wid].tolist()
            if not idx: continue
            r = idx[0]
            table.at[r, s.nama_kelas]=entry
    table.to_csv(os.path.join(gen_dir,"combined_timetable.csv"), index=False)
    with open(os.path.join(gen_dir,"fitness.json"), "w", encoding="utf-8") as f:
        json.dump({"best":float(best_f),"avg":float(sum(fitnesses)/len(fitnesses)),"worst":float(min(fitnesses))}, f, indent=2)
    return gen_dir, table

def save_final(run_dir, population, fitnesses, best_idx, sessions, id_to_slot, ordered_slots, tables):
    final_dir = os.path.join(run_dir,"final")
    os.makedirs(final_dir, exist_ok=True)
    best = population[best_idx]
    slots_map, guru_map = best
    rows=[]
    for wid in ordered_slots:
        r = id_to_slot[wid]
        rows.append({"id_waktu":wid,"hari":r["hari"],"jam_ke":r["jam_ke"]})
    df_slots = pd.DataFrame(rows)
    class_names = sorted({s.nama_kelas for s in sessions})
    table = pd.DataFrame(index=range(len(df_slots)), columns=["hari","jam_ke"]+class_names)
    table["hari"]=df_slots["hari"]; table["jam_ke"]=df_slots["jam_ke"]
    table.fillna("", inplace=True)
    guru_df = tables["guru"]
    for s in sessions:
        st = slots_map.get(s.sid)
        g = guru_map.get(s.sid)
        if st is None or g is None: continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block: continue
        try:
            nama_g = guru_df.loc[guru_df["id_guru"]==g,"nama_guru"].iloc[0]
        except:
            nama_g = str(g)
        entry = f"{nama_g} - {s.nama_mapel}"
        for wid in block:
            idx = df_slots.index[df_slots["id_waktu"]==wid].tolist()
            if not idx: continue
            r = idx[0]
            table.at[r, s.nama_kelas]=entry
    table.to_csv(os.path.join(final_dir,"combined_timetable.csv"), index=False)
    with open(os.path.join(final_dir,"fitness.json"), "w", encoding="utf-8") as f:
        json.dump({"best":float(fitnesses[best_idx]),"avg":float(sum(fitnesses)/len(fitnesses)),"worst":float(min(fitnesses))}, f, indent=2)
    return final_dir, table

# ========== Save to DB ==========
def save_to_db(best, sessions, id_to_slot):
    slots_map, guru_map = best
    tahun = input("Tahun ajaran (e.g. 2025/2026): ").strip()
    semester = input("Semester (ganjil/genap): ").strip()
    keterangan = input("Keterangan (opsional): ").strip()
    with engine.begin() as conn:
        res = conn.execute(text("INSERT INTO jadwal_master (tahun_ajaran, semester, keterangan, dibuat_pada) VALUES (:t,:s,:k,CURRENT_TIMESTAMP())"),
                           {"t":tahun,"s":semester,"k":keterangan})
        master_id = res.lastrowid
        rows=[]
        for s in sessions:
            st = slots_map.get(s.sid)
            g = guru_map.get(s.sid)
            if st is None or g is None: continue
            block = find_consecutive(st, s.length, id_to_slot, day_ordered)
            if not block: continue
            for wid in block:
                rows.append({"id_master":master_id,"id_kelas":s.id_kelas,"id_mapel":s.id_mapel,"id_guru":g,"id_ruang":None,"id_waktu":wid,"generasi":None,"fitness":None})
        if rows:
            df = pd.DataFrame(rows)
            df.to_sql("jadwal", con=conn, if_exists="append", index=False)
    print(f"[DB] Saved jadwal_master id {master_id} with {len(rows)} row(s).")
    return master_id

# ========== MAIN ==========
if __name__ == "__main__":
    print("GA Global - HARD (session-aware) starting...")
    tables = read_tables()
    id_to_slot, day_ordered, ordered_slots = build_slots(tables["waktu"])
    sessions, expected_per_class = build_sessions(tables, target_hours_per_class=40)
    candidates = build_candidates(tables["guru_mapel"])
    guru_capacity = {int(r["id_guru"]): int(r["jam_mingguan"]) for _,r in tables["guru"].iterrows()}

    print(f"[info] sessions count: {len(sessions)}")
    total_required = sum(expected_per_class.values())
    print(f"[info] required total slots (sum classes): {total_required}")
    print(f"[info] available slots: {len(ordered_slots)}")
    if total_required > len(ordered_slots):
        print("[WARN] Required total slots > available slots — impossible to satisfy all. Check waktu table or target hours.")

    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT, f"run_{run_label}")
    os.makedirs(run_dir, exist_ok=True)
    # snapshot
    for k,v in tables.items():
        try: v.to_csv(os.path.join(run_dir,f"{k}.csv"), index=False)
        except: pass

    # population
    print("[info] creating initial population...")
    population = create_initial_population(sessions, id_to_slot, day_ordered, candidates, POP_SIZE)
    fitnesses=[]
    for ind in population:
        f,diag = fitness_hard(ind, sessions, id_to_slot, day_ordered, ordered_slots, expected_per_class, candidates, guru_capacity)
        fitnesses.append(f)
    print(f"[info] initial avg={sum(fitnesses)/len(fitnesses):.8f} best={max(fitnesses):.8f}")

    best_overall=None; best_overall_f=-1
    for gen in range(1, GENERATIONS+1):
        avg = sum(fitnesses)/len(fitnesses)
        best_f = max(fitnesses); worst_f = min(fitnesses)
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        print(f"[Gen {gen:03d}] avg={avg:.8f} best={best_f:.8f} worst={worst_f:.8f}")
        # save
        gen_dir, _ = save_generation(run_dir, gen, population, fitnesses, best_idx, sessions, id_to_slot, ordered_slots, tables)
        # update best overall
        if best_f > best_overall_f:
            best_overall_f = best_f
            best_overall = deepcopy(population[best_idx])
        # next population
        new_pop = [deepcopy(population[best_idx])]  # elitism keep best1
        while len(new_pop) < POP_SIZE:
            p1 = tournament(population, fitnesses, k=3)
            p2 = tournament(population, fitnesses, k=3)
            c1,c2 = chunk_safe_crossover(p1,p2,sessions,prob=0.85)
            c1 = chunk_safe_mutation(c1,sessions,id_to_slot,day_ordered,candidates,prob=0.12)
            c2 = chunk_safe_mutation(c2,sessions,id_to_slot,day_ordered,candidates,prob=0.12)
            c1 = repair_light(c1,sessions,id_to_slot,day_ordered,candidates)
            c2 = repair_light(c2,sessions,id_to_slot,day_ordered,candidates)
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)
        population = new_pop
        fitnesses=[]
        for ind in population:
            f,diag = fitness_hard(ind, sessions, id_to_slot, day_ordered, ordered_slots, expected_per_class, candidates, guru_capacity)
            fitnesses.append(f)

    # finish
    print("GA finished.")
    final_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    final_dir, final_table = save_final(run_dir, population, fitnesses, final_idx, sessions, id_to_slot, ordered_slots, tables)
    print(f"[INFO] Final saved to {final_dir}")
    # preview small
    print(final_table.fillna("").head(30).to_string(index=False))
    ans = input("Simpan final ke DB jadwal_master/jadwal? (y/n): ").strip().lower()
    if ans == "y":
        save_to_db(population[final_idx], sessions, id_to_slot)
    print("Done.")
