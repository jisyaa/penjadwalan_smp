#!/usr/bin/env python3
"""
ga_global_scheduler_hard.py

Global Genetic Algorithm — HARD constraint mode (sessions must be consecutive).
- 1 individual = full-week schedule for all classes.
- Every generation saved under results/run_<timestamp>/gen_XXX/
- Final best saved under results/run_<timestamp>/final/
- Uses DB schema provided by user (tables: guru, mapel, kelas, guru_mapel, waktu, jadwal, jadwal_master).
- Enforces 40 jam/minggu per class (expected) and chunking rules.

Usage:
    python ga_global_scheduler_hard.py

Start with small POP_SIZE / GENERATIONS for testing.

Author: Assistant (adapted)
"""
import os, random, json, math, time
from datetime import datetime
from collections import defaultdict, namedtuple
from copy import deepcopy
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ---------------- Configuration ----------------
DB_USER = os.getenv("DB_USER","root")
DB_PASS = os.getenv("DB_PASS","")
DB_HOST = os.getenv("DB_HOST","127.0.0.1")
DB_PORT = os.getenv("DB_PORT","3306")
DB_NAME = os.getenv("DB_NAME","db_penjadwalan")

RESULTS_ROOT = os.getenv("RESULTS_DIR","./results")
POP_SIZE = int(os.getenv("POP_SIZE","20"))
GENERATIONS = int(os.getenv("GENERATIONS","60"))

CONN_STR = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONN_STR, pool_pre_ping=True)

random.seed(42)

# ---------------- Data structures ----------------
Session = namedtuple("Session", ["sid","id_kelas","nama_kelas","id_mapel","nama_mapel","length","split_group"])

# ---------------- DB Read ----------------
def read_tables():
    with engine.connect() as conn:
        tables = {}
        for t in ["guru","mapel","kelas","guru_mapel","waktu"]:
            tables[t] = pd.read_sql(text(f"SELECT * FROM {t}"), conn)
    return tables

# ---------------- Build available slots ----------------
def build_slots_from_db(waktu_df):
    # Respect DB: only slots with jam_ke NOT NULL and keterangan null/empty are usable
    df = waktu_df.copy()
    avail = df[(df["jam_ke"].notnull()) & ((df["keterangan"].isnull()) | (df["keterangan"].astype(str).str.strip()==""))]
    avail = avail.sort_values(["hari","jam_ke"])
    id_to_slot = {int(r["id_waktu"]): r for _, r in avail.iterrows()}
    day_ordered = defaultdict(list)
    for _, r in avail.iterrows():
        day_ordered[r["hari"]].append(int(r["id_waktu"]))
    # sort each day
    for d in day_ordered:
        day_ordered[d].sort(key=lambda wid: id_to_slot[wid]["jam_ke"])
    # ordered_slots list (Senin..Jumat natural)
    day_order = ["Senin","Selasa","Rabu","Kamis","Jumat"]
    ordered_slots = []
    for d in day_order:
        if d in day_ordered:
            ordered_slots.extend(day_ordered[d])
    return id_to_slot, day_ordered, ordered_slots

# ---------------- Session creation (global) ----------------
def build_sessions_global(tables, expected_hours_per_class=40):
    """
    Create sessions per rules:
     - 1 -> [1]
     - 2 -> [2]
     - 3 -> [3]
     - 4 -> [2,2]
     - 5 -> [3,2]
     - 6 -> [3,3]
    Use only guru_mapel aktif entries to produce class-mapel pairs.
    Return sessions list and expected_per_class dict (nama_kelas -> expected hours).
    """
    gm = tables["guru_mapel"][tables["guru_mapel"]["aktif"]=="aktif"]
    mapel_df = tables["mapel"]
    kelas_df = tables["kelas"]
    sessions = []
    sid = 1
    split_gid = 1
    expected_per_class = defaultdict(int)
    pairs = gm.groupby(["id_kelas","id_mapel"]).size().reset_index()[["id_kelas","id_mapel"]]
    for _, r in pairs.iterrows():
        id_kelas = int(r["id_kelas"])
        id_mapel = int(r["id_mapel"])
        kelas_row = kelas_df[kelas_df["id_kelas"]==id_kelas].iloc[0]
        mapel_row = mapel_df[mapel_df["id_mapel"]==id_mapel].iloc[0]
        nama_kelas = kelas_row["nama_kelas"]
        nama_mapel = mapel_row["nama_mapel"]
        jam = int(mapel_row["jam_per_minggu"])
        # Map rule:
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
        expected_per_class[nama_kelas] += jam
    # Override expected hours per class to the policy (if user wants 40 fixed)
    # User confirmed 40 hours per class; ensure expected_per_class reflect that:
    for k in expected_per_class:
        expected_per_class[k] = 40
    return sessions, expected_per_class

# ---------------- Candidate map ----------------
def build_candidates_map(guru_mapel_df):
    cand = defaultdict(list)
    for _, r in guru_mapel_df[guru_mapel_df["aktif"]=="aktif"].iterrows():
        cand[(int(r["id_kelas"]), int(r["id_mapel"]))].append(int(r["id_guru"]))
    return cand

# ---------------- Utilities ----------------
def find_consecutive(start_wid, length, id_to_slot, day_ordered):
    if start_wid not in id_to_slot: return None
    day = id_to_slot[start_wid]["hari"]
    ordered = day_ordered[day]
    try:
        idx = ordered.index(start_wid)
    except ValueError:
        return None
    block = ordered[idx: idx+length]
    if len(block) != length: return None
    prev = None
    for wid in block:
        jk = id_to_slot[wid]["jam_ke"]
        if prev is not None and jk != prev+1:
            return None
        prev = jk
    return block

# ---------------- Initial population (session-based global) ----------------
def create_initial_population(sessions, id_to_slot, day_ordered, candidates_map, pop_size):
    """
    Create feasible-ish individuals by greedy packing per class, trying to avoid gap,
    but also randomizing day order to create diversity.
    """
    class_sessions = defaultdict(list)
    for s in sessions:
        class_sessions[s.nama_kelas].append(s)
    # sort sessions per class by descending length
    for k in class_sessions:
        class_sessions[k].sort(key=lambda x: (-x.length, x.split_group or 0))
    population = []
    for _ in range(pop_size):
        slots_map = {}
        guru_map = {}
        busy = {}  # (guru,wid) -> True
        class_assigned_slots = defaultdict(set)
        # order classes randomized
        classes = list(class_sessions.keys())
        random.shuffle(classes)
        for cls in classes:
            sess_list = class_sessions[cls]
            # attempt to pack per day using natural day order
            days = ["Senin","Selasa","Rabu","Kamis","Jumat"]
            for s in sess_list:
                placed=False
                # choose day preference shuffle but deterministic order used first
                day_choices = days.copy()
                random.shuffle(day_choices)
                # try each day for block
                for d in day_choices:
                    ordered = day_ordered.get(d,[])
                    for i in range(len(ordered)):
                        start = ordered[i]
                        block = find_consecutive(start,s.length,id_to_slot,day_ordered)
                        if not block: continue
                        # ensure block doesn't create gap for class on that day:
                        # allow placement if either empty on that day or fits contiguous with existing
                        existing = [id_to_slot[w]["jam_ke"] for w in class_assigned_slots[cls] if id_to_slot[w]["hari"]==d]
                        # if existing non-empty, we ensure no created internal gap: we permit any insertion (repair later)
                        # check teacher availability
                        cands = candidates_map.get((s.id_kelas,s.id_mapel),[])
                        random.shuffle(cands)
                        for g in cands:
                            conflict=False
                            for wid in block:
                                if busy.get((g,wid),False): conflict=True; break
                            if conflict: continue
                            # assign
                            slots_map[s.sid]=block[0]
                            guru_map[s.sid]=g
                            for wid in block:
                                busy[(g,wid)]=True
                                class_assigned_slots[cls].add(wid)
                            placed=True
                            break
                        if placed: break
                    if placed: break
                if not placed:
                    # leave unassigned for repair
                    slots_map[s.sid]=None; guru_map[s.sid]=None
        population.append((slots_map,guru_map))
    return population

# ---------------- Fitness (hard-mode) ----------------
def evaluate_global(chrom, sessions, id_to_slot, day_ordered, ordered_slots, expected_per_class, candidates_map, guru_capacity):
    slots_map, guru_map = chrom
    penalty = 0
    details = defaultdict(int)
    teacher_time = defaultdict(list)
    class_time = defaultdict(list)
    assigned_per_class = defaultdict(set)
    split_days = defaultdict(set)

    # expand assignments
    for s in sessions:
        start = slots_map.get(s.sid)
        g = guru_map.get(s.sid)
        if start is None or g is None:
            penalty += 1_000_000  # heavy
            details["missing"] += 1
            continue
        block = find_consecutive(start, s.length, id_to_slot, day_ordered)
        if not block:
            penalty += 1_000_000
            details["non_consec"] += 1
            continue
        # ensure none of block slots have keterangan (we already filtered slots but double-check)
        for wid in block:
            row = id_to_slot.get(wid)
            if row is None:
                penalty += 1_000_000; details["invalid_slot"]+=1
            else:
                if row["keterangan"] is not None and str(row["keterangan"]).strip()!="":
                    penalty += 1_000_000; details["blocked_slot"]+=1
        # register
        for wid in block:
            teacher_time[(g,wid)].append(s.sid)
            class_time[(s.nama_kelas,wid)].append(s.sid)
            assigned_per_class[s.nama_kelas].add(wid)
        if s.split_group is not None:
            split_days[s.split_group].add(id_to_slot[block[0]]["hari"])

    # check teacher conflicts
    for (g,w), lst in teacher_time.items():
        if len(lst) > 1:
            cnt = len(lst)-1
            penalty += 1_000_000 * cnt
            details["teacher_conflicts"] += cnt

    # class conflicts
    for (cls,w), lst in class_time.items():
        if len(lst) > 1:
            cnt = len(lst)-1
            penalty += 1_000_000 * cnt
            details["class_conflicts"] += cnt

    # split group must be on different days
    for gid, days in split_days.items():
        if len(days) < 2:
            penalty += 500_000
            details["split_violation"] += 1

    # gaps per class per day (no internal holes)
    for cls, wids in assigned_per_class.items():
        per_day = defaultdict(list)
        for wid in wids:
            day = id_to_slot[wid]["hari"]
            per_day[day].append(id_to_slot[wid]["jam_ke"])
        for day,jks in per_day.items():
            jks_sorted = sorted(jks)
            for i in range(len(jks_sorted)-1):
                if jks_sorted[i+1] != jks_sorted[i] + 1:
                    # internal gap: penalize heavily (hard constraint)
                    penalty += 200_000
                    details["gaps"] += 1

    # expected hours per class (user set to 40). Penalize missing strongly.
    for cls, expected in expected_per_class.items():
        assigned = len(assigned_per_class.get(cls,set()))
        if assigned < expected:
            diff = expected - assigned
            penalty += 200_000 * diff
            details["missing_hours"] += diff
        elif assigned > expected:
            penalty += 50_000 * (assigned - expected)
            details["over_assign"] += max(0,assigned-expected)

    # teacher overload
    teacher_hours = defaultdict(int)
    for (g,w), lst in teacher_time.items():
        teacher_hours[g] += len(lst)
    for g, hrs in teacher_hours.items():
        cap = guru_capacity.get(g, 9999)
        if hrs > cap:
            extra = hrs-cap
            penalty += 1000 * extra
            details["teacher_overload"] += extra

    fitness = 1.0 / (1.0 + penalty)
    return fitness, penalty, dict(details)

# ---------------- Genetic operators ----------------
def tournament(pop, fitnesses, k=3):
    pick = random.sample(range(len(pop)), min(k,len(pop)))
    best = max(pick, key=lambda i: fitnesses[i])
    return deepcopy(pop[best])

def uniform_crossover(p1, p2, sessions, prob=0.85):
    if random.random() > prob:
        return deepcopy(p1), deepcopy(p2)
    sids = [s.sid for s in sessions]
    a_slots, a_gurus = deepcopy(p1[0]), deepcopy(p1[1])
    b_slots, b_gurus = deepcopy(p2[0]), deepcopy(p2[1])
    for sid in sids:
        if random.random() < 0.5:
            a_slots[sid], b_slots[sid] = b_slots.get(sid), a_slots.get(sid)
            a_gurus[sid], b_gurus[sid] = b_gurus.get(sid), a_gurus.get(sid)
    return (a_slots,a_gurus),(b_slots,b_gurus)

def mutation(ind, sessions, id_to_slot, day_ordered, candidates_map, prob=0.12):
    slots_map, guru_map = deepcopy(ind[0]), deepcopy(ind[1])
    for s in sessions:
        if random.random() < prob:
            # change start to another valid start
            valid_starts = []
            for d, ordered in day_ordered.items():
                for i in range(len(ordered)):
                    st = ordered[i]
                    if find_consecutive(st, s.length, id_to_slot, day_ordered):
                        valid_starts.append(st)
            if valid_starts:
                slots_map[s.sid] = random.choice(valid_starts)
        if random.random() < prob:
            cands = candidates_map.get((s.id_kelas, s.id_mapel), [])
            if cands:
                guru_map[s.sid] = random.choice(cands)
    return (slots_map, guru_map)

# ---------------- Lightweight repair ----------------
def repair(ind, sessions, id_to_slot, day_ordered, candidates_map):
    # Attempt to fill missing sessions by scanning for free blocks ignoring global teacher conflicts
    slots_map, guru_map = deepcopy(ind[0]), deepcopy(ind[1])
    # compute per-class assigned slots to avoid class double-book
    class_assigned = defaultdict(set)
    for s in sessions:
        st = slots_map.get(s.sid)
        g = guru_map.get(s.sid)
        if st is None or g is None: continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block:
            slots_map[s.sid]=None; guru_map[s.sid]=None; continue
        for wid in block:
            class_assigned[s.nama_kelas].add(wid)
    # try to place unassigned sessions (greedy)
    for s in [ss for ss in sessions if slots_map.get(ss.sid) is None]:
        placed=False
        # try days in order
        for day in ["Senin","Selasa","Rabu","Kamis","Jumat"]:
            ordered = day_ordered.get(day, [])
            for i in range(len(ordered)):
                st = ordered[i]
                block = find_consecutive(st, s.length, id_to_slot, day_ordered)
                if not block: continue
                if any(w in class_assigned[s.nama_kelas] for w in block): continue
                cands = candidates_map.get((s.id_kelas,s.id_mapel), [])
                for g in cands:
                    # accept (we ignore teacher global conflicts in repair; strong fitness will punish)
                    slots_map[s.sid]=block[0]; guru_map[s.sid]=g
                    for w in block: class_assigned[s.nama_kelas].add(w)
                    placed=True; break
                if placed: break
            if placed: break
    return (slots_map, guru_map)

# ---------------- Save generation & final ----------------
def save_generation(run_dir, gen_idx, population, fitnesses, best_idx, sessions, id_to_slot, ordered_slots, tables):
    gen_dir = os.path.join(run_dir, f"gen_{gen_idx:03d}")
    os.makedirs(gen_dir, exist_ok=True)
    best = population[best_idx]
    best_f = fitnesses[best_idx]
    # build combined table (Format B)
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
    # fill
    slots_map, guru_map = best
    for s in sessions:
        st = slots_map.get(s.sid)
        g = guru_map.get(s.sid)
        if st is None or g is None: continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block: continue
        try:
            nama_g = guru_df.loc[guru_df["id_guru"]==g,"nama_guru"].iloc[0]
        except Exception:
            nama_g = str(g)
        entry = f"{nama_g} - {s.nama_mapel}"
        for wid in block:
            idx = df_slots.index[df_slots["id_waktu"]==wid].tolist()
            if not idx: continue
            r = idx[0]
            table.at[r, s.nama_kelas] = entry
    # save CSV and fitness json
    table.to_csv(os.path.join(gen_dir,"combined_timetable.csv"), index=False)
    with open(os.path.join(gen_dir,"fitness.json"),"w",encoding="utf-8") as f:
        json.dump({"best":float(best_f),"avg":float(sum(fitnesses)/len(fitnesses)),"worst":float(min(fitnesses))}, f, indent=2)
    return gen_dir, table

def save_final(run_dir, population, fitnesses, best_idx, sessions, id_to_slot, ordered_slots, tables):
    final_dir = os.path.join(run_dir,"final")
    os.makedirs(final_dir, exist_ok=True)
    # reuse save_generation logic but put under final
    best = population[best_idx]
    best_f = fitnesses[best_idx]
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
    slots_map,guru_map = best
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
            table.at[r, s.nama_kelas] = entry
    table.to_csv(os.path.join(final_dir,"combined_timetable.csv"), index=False)
    with open(os.path.join(final_dir,"fitness.json"),"w",encoding="utf-8") as f:
        json.dump({"best":float(best_f),"avg":float(sum(fitnesses)/len(fitnesses)),"worst":float(min(fitnesses))}, f, indent=2)
    return final_dir, table

# ---------------- Save best to DB ----------------
def save_best_to_db(best, sessions, id_to_slot, tables):
    slots_map,guru_map = best
    tahun = input("Masukkan Tahun Ajaran (mis: 2025/2026): ").strip()
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
    print(f"[DB] saved jadwal_master id {master_id} with {len(rows)} rows")
    return master_id

# ---------------- Main GA ----------------
if __name__ == "__main__":
    print("Starting GA Global (HARD) scheduler...")
    tables = read_tables()
    id_to_slot, day_ordered, ordered_slots = build_slots_from_db(tables["waktu"])
    sessions, expected_per_class = build_sessions_global(tables, expected_hours_per_class=40)
    candidates_map = build_candidates_map(tables["guru_mapel"])
    guru_capacity = {int(r["id_guru"]): int(r["jam_mingguan"]) for _,r in tables["guru"].iterrows()}

    total_required = sum(expected_per_class.values())
    total_available = len(ordered_slots)
    print(f"[info] Sessions to place: {len(sessions)}")
    print(f"[info] Required hours (sum classes): {total_required}, Available slots: {total_available}")

    # quick feasibility check
    if total_required > total_available:
        print("[WARN] Required total hours exceed total available slots. Hard constraints may make solution impossible.")
        # but still proceed; GA will show impossibility by never reaching perfect fitness

    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT, f"run_{run_label}")
    os.makedirs(run_dir, exist_ok=True)
    # snapshot tables
    for k,v in tables.items():
        try: v.to_csv(os.path.join(run_dir,f"{k}.csv"), index=False)
        except: pass

    # initial population
    print("[info] Generating initial population...")
    population = create_initial_population(sessions, id_to_slot, day_ordered, candidates_map, POP_SIZE)
    # evaluate
    fitnesses = []
    for ind in population:
        f,pen,det = evaluate_global(ind, sessions, id_to_slot, day_ordered, ordered_slots, expected_per_class, candidates_map, guru_capacity)
        fitnesses.append(f)
    print(f"[info] Init avg fitness: {sum(fitnesses)/len(fitnesses):.8f}, best: {max(fitnesses):.8f}")

    best_overall = None; best_f_overall = -1
    # GA loop
    for gen in range(1, GENERATIONS+1):
        # report
        avg_f = sum(fitnesses)/len(fitnesses)
        best_f = max(fitnesses); worst_f = min(fitnesses)
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        print(f"[Gen {gen:03d}] avg={avg_f:.8f} best={best_f:.8f} worst={worst_f:.8f}")
        # save generation
        gen_dir, _ = save_generation(run_dir, gen, population, fitnesses, best_idx, sessions, id_to_slot, ordered_slots, tables)
        # update best overall
        if best_f > best_f_overall:
            best_f_overall = best_f
            best_overall = deepcopy(population[best_idx])
        # produce next population with elitism
        new_pop = [deepcopy(population[best_idx])]  # keep best 1
        while len(new_pop) < POP_SIZE:
            p1 = tournament(population, fitnesses, k=3)
            p2 = tournament(population, fitnesses, k=3)
            c1, c2 = uniform_crossover(p1,p2,sessions,prob=0.85)
            c1 = mutation(c1,sessions,id_to_slot,day_ordered,candidates_map,prob=0.12)
            c2 = mutation(c2,sessions,id_to_slot,day_ordered,candidates_map,prob=0.12)
            c1 = repair(c1,sessions,id_to_slot,day_ordered,candidates_map)
            c2 = repair(c2,sessions,id_to_slot,day_ordered,candidates_map)
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)
        population = new_pop
        fitnesses = []
        for ind in population:
            f,pen,det = evaluate_global(ind, sessions, id_to_slot, day_ordered, ordered_slots, expected_per_class, candidates_map, guru_capacity)
            fitnesses.append(f)

    # After GA
    print("GA finished.")
    # Save final best into final/
    final_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    final_dir, final_table = save_final(run_dir, population, fitnesses, final_best_idx, sessions, id_to_slot, ordered_slots, tables)
    print(f"[INFO] Final timetable saved to {final_dir}")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(final_table.fillna("").to_string(index=False))
    # ask to save to DB
    ans = input("Simpan jadwal final ke database jadwal_master/jadwal? (y/n): ").strip().lower()
    if ans == "y":
        save_best_to_db(population[final_best_idx], sessions, id_to_slot, tables)
    print("Done.")
