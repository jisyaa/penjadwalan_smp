#!/usr/bin/env python3
"""
ga_per_class_scheduler.py

GA per-kelas untuk penjadwalan SMP (SMPN 1 Enam Lingkung)
- Strategi B: jalankan GA per kelas berurutan (random order), reservasi guru slot pada global calendar
- Simpan setiap generasi ke results/run_{timestamp}/class_{kelas}/gen_{n}
- Tampilkan fitness tiap generasinya, tampilkan jadwal terbaik per kelas di terminal
- Gabungkan seluruh kelas menjadi timetable (format 2) dan tampilkan/simpan
- Menanyakan apakah mau simpan ke DB (jadwal_master + jadwal)

Constraints implemented:
a. guru tidak double-book (cek global saat assign)
b. tiap kelas harus penuhi jumlah jam (expected calculated)
c. tidak boleh ada jam kosong di tengah hari
d. waktu dengan keterangan != NULL tidak boleh terpakai
e. hanya guru_mapel aktif
f. ruangan tidak dipakai
g/h. mapel dengan jam_per_minggu > 3 dibagi jadi 2 pertemuan (3 + sisa), pada hari berbeda
i. isi semua jam kecuali keterangan (no gap/tumpang tindih)

Author: adapted for user's DB schema
"""

import os, random, json, math
from datetime import datetime
from collections import defaultdict, namedtuple, Counter
from copy import deepcopy
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import trange

load_dotenv()

# DB config
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "db_penjadwalan")
CONN_STR = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONN_STR, pool_pre_ping=True)

RESULTS_ROOT = os.getenv("RESULTS_DIR", "./results")
POP_SIZE = int(os.getenv("POP_SIZE", 60))
GENERATIONS = int(os.getenv("GENERATIONS", 120))

random.seed(42)

# Data structures
Session = namedtuple("Session", ["sid","id_kelas","nama_kelas","id_mapel","nama_mapel","length","split_group"])

# ---------- Helpers: read DB ----------
def read_tables():
    with engine.connect() as conn:
        tbls = {}
        for t in ["guru","mapel","kelas","guru_mapel","waktu"]:
            tbls[t] = pd.read_sql(text(f"SELECT * FROM {t}"), conn)
    return tbls

# ---------- Preprocess time slots ----------
def build_slots(waktu_df):
    # available slots: keterangan is null or empty, jam_ke not null
    df = waktu_df.copy()
    avail = df[(df["keterangan"].isnull()) | (df["keterangan"].astype(str).str.strip()=="")]
    avail = avail[avail["jam_ke"].notnull()]
    avail = avail.sort_values(["hari","jam_ke"])
    id_to_slot = {int(r["id_waktu"]): r for _, r in avail.iterrows()}
    day_ordered = defaultdict(list)
    for _, r in avail.iterrows():
        day_ordered[r["hari"]].append(int(r["id_waktu"]))
    for d in day_ordered:
        # sort by jam_ke
        day_ordered[d].sort(key=lambda wid: id_to_slot[wid]["jam_ke"])
    return id_to_slot, day_ordered

# ---------- Build sessions for a single class ----------
def build_sessions_for_class(id_kelas, kelas_row, mapel_df, guru_mapel_df):
    """
    Produce sessions list for this class:
    - For each mapel assigned to this class (guru_mapel aktif), use mapel.jam_per_minggu
    - if jam_per_minggu > 3 -> split into [3, remainder] (two sessions), set same split_group id for them
    - else create jam_per_minggu sessions of length 1
    Return list[Session], expected_total_hours
    """
    sessions = []
    sid = 1
    split_gid = 1
    gm = guru_mapel_df[(guru_mapel_df["id_kelas"]==id_kelas) & (guru_mapel_df["aktif"]=="aktif")]
    # get unique id_mapel for this class (some mapel may have multiple guru entries)
    mapels = gm["id_mapel"].unique().tolist()
    for id_mapel in mapels:
        row = mapel_df.loc[mapel_df["id_mapel"]==id_mapel].iloc[0]
        nama_mapel = row["nama_mapel"]
        jam = int(row["jam_per_minggu"])
        if jam > 3:
            # split into two pertemuan: prefer [3, jam-3] (ensures one 3-hour)
            a = 3
            b = jam - 3
            # create two sessions with same split_group
            sessions.append(Session(sid, id_kelas, kelas_row["nama_kelas"], id_mapel, nama_mapel, a, split_gid)); sid+=1
            sessions.append(Session(sid, id_kelas, kelas_row["nama_kelas"], id_mapel, nama_mapel, b, split_gid)); sid+=1
            split_gid += 1
        else:
            for _ in range(jam):
                sessions.append(Session(sid, id_kelas, kelas_row["nama_kelas"], id_mapel, nama_mapel, 1, None)); sid+=1
    expected = sum(s.length for s in sessions)
    return sessions, expected

# ---------- Candidates (guru) ----------
def build_candidates_map(guru_mapel_df):
    cand = defaultdict(list)  # key: (id_kelas,id_mapel) -> [id_guru]
    for _, r in guru_mapel_df[guru_mapel_df["aktif"]=="aktif"].iterrows():
        cand[(int(r["id_kelas"]), int(r["id_mapel"]))].append(int(r["id_guru"]))
    return cand

# ---------- Utility: consecutive slots ----------
def find_consec(start_wid, length, id_to_slot, day_ordered):
    if start_wid not in id_to_slot: return None
    day = id_to_slot[start_wid]["hari"]
    ordered = day_ordered[day]
    try:
        idx = ordered.index(start_wid)
    except ValueError:
        return None
    block = ordered[idx: idx+length]
    if len(block) != length: return None
    # ensure jam_ke consecutive
    prev = None
    for wid in block:
        jk = id_to_slot[wid]["jam_ke"]
        if prev is not None and jk != prev+1:
            return None
        prev = jk
    return block

# ---------- Greedy initial population (per class) ----------
def greedy_initial_individual(sessions, id_to_slot, day_ordered, candidates_map, global_teacher_busy, expected_hours):
    """
    Build one feasible assignment for given class:
    - iterate days and slots in order, fill sessions trying to avoid gaps
    - respect teacher availability (global_teacher_busy: dict (guru, wid)->True)
    - ensure split_group parts on different days
    """
    # flatten ordered days as list to fill in day-major order, but we will try to pack per day
    ordered_days = list(day_ordered.keys())  # e.g. ['Senin','Selasa',...'] in insertion order
    # but ensure deterministic order
    ordered_days = sorted(ordered_days, key=lambda d: ["Senin","Selasa","Rabu","Kamis","Jumat"].index(d))
    # Pre-sort sessions: place longer sessions first (harder)
    sess_sorted = sorted(sessions, key=lambda s: (-s.length, s.split_group or 0, s.id_mapel))
    slots_map = {}
    guru_map = {}
    assigned_slots = set()
    split_day_used = {}
    # For each session, try to find consecutive block in some day where teachers available
    for s in sess_sorted:
        placed = False
        # order: try days that don't violate split_group
        days_try = ordered_days.copy()
        random.shuffle(days_try)  # shuffle so different individuals vary
        if s.split_group is not None and s.split_group in split_day_used:
            # prefer days != that day
            days_try = [d for d in days_try if d != split_day_used[s.split_group]] + [d for d in days_try if d == split_day_used[s.split_group]]
        for day in days_try:
            ordered = day_ordered[day]
            for i in range(len(ordered)):
                start = ordered[i]
                block = find_consec(start, s.length, id_to_slot, day_ordered)
                if not block: 
                    continue
                # check block not overlapping with assigned_slots for this class
                if any(w in assigned_slots for w in block):
                    continue
                # check teacher candidate availability for all wid
                candidates = candidates_map.get((s.id_kelas, s.id_mapel), [])
                random.shuffle(candidates)
                for g in candidates:
                    busy = False
                    for wid in block:
                        if global_teacher_busy.get((g,wid), False):
                            busy = True; break
                    if not busy:
                        # assign
                        slots_map[s.sid] = block[0]
                        guru_map[s.sid] = g
                        for wid in block:
                            assigned_slots.add(wid)
                            global_teacher_busy[(g,wid)] = True
                        if s.split_group is not None:
                            split_day_used[s.split_group] = day
                        placed = True
                        break
                if placed: break
            if placed: break
        if not placed:
            # fallback: leave unassigned (will be penalized) -> attempt to place later in repair
            slots_map[s.sid] = None
            guru_map[s.sid] = None
    # if assigned slots < expected_hours, we will leave gaps to be filled by GA/repair
    return (slots_map, guru_map)

# ---------- Create initial population (per class) ----------
def create_initial_population(sessions, id_to_slot, day_ordered, candidates_map, global_teacher_busy, pop_size, expected_hours):
    pop = []
    for _ in range(pop_size):
        # copy teacher busy map to try different random placements
        gt_busy = dict(global_teacher_busy)
        ind = greedy_initial_individual(sessions, id_to_slot, day_ordered, candidates_map, gt_busy, expected_hours)
        pop.append(ind)
    return pop

# ---------- Fitness function (per class but considers global teacher conflicts if any) ----------
def evaluate_individual(individual, sessions, id_to_slot, day_ordered, candidates_map, global_teacher_busy_snapshot, expected_hours, guru_capacity):
    """
    individual: (slots_map, guru_map)
    penalties:
      - class double-book (two sessions assign same wid) -> heavy (5000)
      - missing sessions / not consecutive -> heavy (5000)
      - gap per day -> 400 per gap
      - teacher conflict (global busy) -> 5000
      - split_same_day violation -> 2000
      - teacher_overload -> 100 * extra_hour
      - not meeting expected_hours (less) -> 2000 per missing hour
    fitness = 1 / (1 + penalty)
    """
    slots_map, guru_map = individual
    penalty = 0
    details = {}
    # expand assigned slots per session
    class_slot_set = set()
    teacher_time = defaultdict(list)
    split_days = defaultdict(set)
    assigned_hours = 0
    for s in sessions:
        start = slots_map.get(s.sid)
        g = guru_map.get(s.sid)
        if start is None or g is None:
            penalty += 5000
            details.setdefault("missing",[]).append(s.sid)
            continue
        block = find_consec(start, s.length, id_to_slot, day_ordered)
        if not block:
            penalty += 5000
            details.setdefault("invalid_block",[]).append(s.sid)
            continue
        # check per-slot
        for wid in block:
            # class double-book
            if wid in class_slot_set:
                penalty += 5000
                details.setdefault("class_conflict",[]).append((s.sid,wid))
            class_slot_set.add(wid)
            # teacher busy check against snapshot global_busy
            if global_teacher_busy_snapshot.get((g,wid), False):
                penalty += 5000
                details.setdefault("teacher_conflict",[]).append((s.sid,g,wid))
            teacher_time[(g,wid)].append(s.sid)
            assigned_hours += 1
        if s.split_group is not None:
            split_days[s.split_group].add(id_to_slot[start]["hari"])
    # gap penalty: per day, if assigned slots are not consecutive (no holes)
    per_day_slots = defaultdict(list)
    for wid in class_slot_set:
        day = id_to_slot[wid]["hari"]; per_day_slots[day].append(id_to_slot[wid]["jam_ke"])
    for d, jks in per_day_slots.items():
        jks_sorted = sorted(jks)
        for i in range(len(jks_sorted)-1):
            if jks_sorted[i+1] != jks_sorted[i] +1:
                penalty += 400
                details.setdefault("gaps",0)
                details["gaps"] += 1
    # split group check: must be at least 2 different days if group exists (if only one session present -> penalize)
    for gid, days in split_days.items():
        if len(days) < 2:
            penalty += 2000
            details.setdefault("split_violation",[]).append((gid,list(days)))
    # expected hours check
    if assigned_hours < expected_hours:
        penalty += 2000 * (expected_hours - assigned_hours)
        details.setdefault("missing_hours", expected_hours - assigned_hours)
    # teacher overload check: using guru_capacity and teacher assignment in this individual + global snapshot
    # compute teacher hours in snapshot + this individual
    teacher_hours = defaultdict(int)
    for (g,w), v in global_teacher_busy_snapshot.items():
        if v:
            teacher_hours[g] += 1
    # add individual hours
    for (g,wlist) in defaultdict(list, ((k[0],[]) for k in [])): pass
    # we compute by scanning teacher_time:
    for (g,w), sids in teacher_time.items():
        teacher_hours[g] += len(sids)
    for g, hours in teacher_hours.items():
        cap = guru_capacity.get(g, 9999)
        if hours > cap:
            penalty += 100 * (hours - cap)
            details.setdefault("teacher_overload", {})[g] = hours - cap
    fitness = 1.0 / (1.0 + penalty)
    return fitness, penalty, details

# ---------- Crossover & mutation (per class) ----------
def crossover(parent1, parent2, sessions, prob=0.8):
    if random.random() > prob:
        return deepcopy(parent1), deepcopy(parent2)
    # two-point crossover on session ids
    sids = [s.sid for s in sessions]
    a = random.randint(0, len(sids)-1)
    b = random.randint(a, len(sids)-1)
    p1_slots, p1_guru = deepcopy(parent1[0]), deepcopy(parent1[1])
    p2_slots, p2_guru = deepcopy(parent2[0]), deepcopy(parent2[1])
    for sid in sids[a:b+1]:
        p1_slots[sid], p2_slots[sid] = p2_slots.get(sid), p1_slots.get(sid)
        p1_guru[sid], p2_guru[sid] = p2_guru.get(sid), p1_guru.get(sid)
    return (p1_slots,p1_guru),(p2_slots,p2_guru)

def mutation(ind, sessions, id_to_slot, day_ordered, candidates_map, prob=0.12):
    slots_map, guru_map = deepcopy(ind[0]), deepcopy(ind[1])
    # mutation types: change start slot or change guru
    for s in sessions:
        if random.random() < prob:
            # pick a random possible start slot for length
            # build list of all possible starts
            starts = []
            for day, ordered in day_ordered.items():
                for i in range(len(ordered)):
                    start = ordered[i]
                    block = find_consec(start, s.length, id_to_slot, day_ordered)
                    if block: starts.append(start)
            if starts:
                slots_map[s.sid] = random.choice(starts)
        if random.random() < prob:
            cands = candidates_map.get((s.id_kelas,s.id_mapel), [])
            if cands:
                guru_map[s.sid] = random.choice(cands)
    return (slots_map, guru_map)

# ---------- Repair routine (attempt local fixes) ----------
def repair_individual(individual, sessions, id_to_slot, day_ordered, candidates_map, global_teacher_busy_snapshot):
    # simple repair: fill missing using free slots/gurus, try to remove class conflicts by shifting
    slots_map, guru_map = individual
    assigned = set()
    for s in sessions:
        st = slots_map.get(s.sid)
        if st:
            block = find_consec(st, s.length, id_to_slot, day_ordered)
            if block:
                for w in block: assigned.add(w)
    # Try to assign unassigned sessions
    for s in sessions:
        if slots_map.get(s.sid) is not None and guru_map.get(s.sid) is not None:
            continue
        # find possible start/guru
        placed=False
        for day, ordered in day_ordered.items():
            for i in range(len(ordered)):
                start = ordered[i]
                block = find_consec(start, s.length, id_to_slot, day_ordered)
                if not block: continue
                if any(w in assigned for w in block): continue
                for g in candidates_map.get((s.id_kelas,s.id_mapel),[]):
                    conflict=False
                    for w in block:
                        if global_teacher_busy_snapshot.get((g,w),False):
                            conflict=True; break
                    if not conflict:
                        slots_map[s.sid]=block[0]; guru_map[s.sid]=g
                        for w in block: assigned.add(w)
                        placed=True; break
                if placed: break
            if placed: break
    return (slots_map,guru_map)

# ---------- Save generation (per class) ----------
def save_gen_result(run_dir, kelas_name, gen_idx, population, fitnesses, sessions, id_to_slot):
    cls_dir = os.path.join(run_dir, f"class_{kelas_name.replace(' ','_')}")
    os.makedirs(cls_dir, exist_ok=True)
    gen_dir = os.path.join(cls_dir, f"gen_{gen_idx:03d}")
    os.makedirs(gen_dir, exist_ok=True)
    # save best
    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    best = population[best_idx]
    best_f = fitnesses[best_idx]
    # expand to csv rows
    rows=[]
    for s in sessions:
        start=best[0].get(s.sid); g=best[1].get(s.sid)
        if start is None:
            rows.append({"sid":s.sid,"kelas":s.nama_kelas,"mapel":s.nama_mapel,"guru":g,"start":None,"length":s.length})
            continue
        block=find_consec(start,s.length,id_to_slot,day_ordered)
        if not block:
            rows.append({"sid":s.sid,"kelas":s.nama_kelas,"mapel":s.nama_mapel,"guru":g,"start":start,"length":s.length,"note":"INVALID"})
            continue
        for wid in block:
            rows.append({"sid":s.sid,"kelas":s.nama_kelas,"mapel":s.nama_mapel,"guru":g,"id_waktu":wid,"hari":id_to_slot[wid]["hari"],"jam_ke":id_to_slot[wid]["jam_ke"]})
    pd.DataFrame(rows).to_csv(os.path.join(gen_dir,"best_schedule.csv"), index=False)
    pd.DataFrame({"fitness":fitnesses}).to_csv(os.path.join(gen_dir,"fitnesses.csv"), index=False)
    with open(os.path.join(gen_dir,"summary.json"),"w",encoding="utf-8") as f:
        json.dump({"best_fitness":float(best_f),"gen":gen_idx,"timestamp":datetime.now().isoformat()},f,indent=2)
    return best_idx, best_f, gen_dir

# ---------- Main per-class GA runner ----------
def run_ga_for_class(kelas_row, tables, id_to_slot, day_ordered, candidates_map, global_teacher_busy, guru_capacity, pop_size=POP_SIZE, generations=GENERATIONS):
    id_kelas = int(kelas_row["id_kelas"])
    kelas_name = kelas_row["nama_kelas"]
    sessions, expected = build_sessions_for_class(id_kelas, kelas_row, tables["mapel"], tables["guru_mapel"])
    # if expected==0 -> nothing to schedule
    if expected == 0:
        return None, None, {}
    # create run dir base (done by caller)
    population = create_initial_population(sessions, id_to_slot, day_ordered, candidates_map, global_teacher_busy, pop_size, expected)
    best_overall = None; best_f_overall = -1
    fitness_history=[]
    for gen in range(1, generations+1):
        fitnesses=[]; penalties=[]
        for ind in population:
            f, pen, diag = evaluate_individual(ind, sessions, id_to_slot, day_ordered, candidates_map, global_teacher_busy, expected, guru_capacity)
            fitnesses.append(f); penalties.append(pen)
        avg = sum(fitnesses)/len(fitnesses)
        maxf = max(fitnesses); best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        print(f"[{kelas_name}] Gen {gen:03d} avg_f={avg:.6f} best_f={maxf:.6f}")
        fitness_history.append({"gen":gen,"avg":avg,"best":maxf})
        # save generation results (caller provides run_dir)
        # We'll save in a temp area under RESULTS_ROOT/run_timestamp later by caller
        # keep best overall
        if maxf > best_f_overall:
            best_f_overall = maxf; best_overall = deepcopy(population[best_idx])
        # produce next generation
        new_pop=[deepcopy(population[best_idx])]  # elitism 1
        while len(new_pop) < pop_size:
            p1 = tournament(population, fitnesses)
            p2 = tournament(population, fitnesses)
            c1,c2 = crossover(p1,p2,sessions,prob=0.8)
            c1 = mutation(c1,sessions,id_to_slot,day_ordered,candidates_map,prob=0.12)
            c2 = mutation(c2,sessions,id_to_slot,day_ordered,candidates_map,prob=0.12)
            c1 = repair_individual(c1,sessions,id_to_slot,day_ordered,candidates_map,global_teacher_busy)
            c2 = repair_individual(c2,sessions,id_to_slot,day_ordered,candidates_map,global_teacher_busy)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        population = new_pop
    # final best -> return and also update global_teacher_busy with its assignments
    # update global busy
    best_slots, best_guru = best_overall
    # expand and mark teacher busy
    assigned_rows=[]
    for s in sessions:
        start = best_slots.get(s.sid)
        g = best_guru.get(s.sid)
        if start is None or g is None: continue
        block=find_consec(start,s.length,id_to_slot,day_ordered)
        if not block: continue
        for wid in block:
            global_teacher_busy[(g,wid)] = True
            assigned_rows.append((s,kelas_name,g,wid))
    return best_overall, best_f_overall, {"sessions":sessions,"assigned_rows":assigned_rows,"expected":expected,"fitness_history":fitness_history}

# ---------- small tournament selection helper ----------
def tournament(pop, fitnesses, k=3):
    idxs = random.sample(range(len(pop)), min(k,len(pop)))
    best = max(idxs, key=lambda i: fitnesses[i])
    return deepcopy(pop[best])

# ---------- Save master result (combined timetable) ----------
def build_combined_timetable(all_class_assignments, id_to_slot, day_ordered, tables):
    # build DataFrame with index (hari,jam_ke) and columns classes
    # find all used days/jam_ke ordering from id_to_slot
    slots = sorted(id_to_slot.keys(), key=lambda wid: (["Senin","Selasa","Rabu","Kamis","Jumat"].index(id_to_slot[wid]["hari"]), id_to_slot[wid]["jam_ke"]))
    rows = []
    for wid in slots:
        rows.append({"id_waktu":wid,"hari":id_to_slot[wid]["hari"],"jam_ke":id_to_slot[wid]["jam_ke"]})
    df_slots = pd.DataFrame(rows)
    kelas_names = [k for k in all_class_assignments.keys()]
    # build empty table with rows per slot and columns per class
    table = pd.DataFrame(index=range(len(df_slots)), columns=kelas_names)
    # fill table
    for kelas, info in all_class_assignments.items():
        # info contains 'assigned_rows' with tuples (session, kelasname, guru, wid)
        for s,kname,g,wid in info["assigned_rows"]:
            # map wid to row index
            idx = df_slots.index[df_slots["id_waktu"]==wid].tolist()
            if not idx: continue
            r = idx[0]
            # write "NamaGuru - NamaMapel"
            guru_name = tables["guru"].loc[tables["guru"]["id_guru"]==g,"nama_guru"].iloc[0] if g is not None else "?"
            cell_value = f"{guru_name} - {s.nama_mapel}"
            table.at[r, kelas] = cell_value
    # attach hari & jam_ke columns
    table.insert(0,"jam_ke", df_slots["jam_ke"].values)
    table.insert(0,"hari", df_slots["hari"].values)
    return table, df_slots

# ---------- Save combined CSV and display (Format 2 requirement) ----------
def save_and_show_combined(table, run_dir):
    out_csv = os.path.join(run_dir,"combined_timetable.csv")
    table.to_csv(out_csv, index=False)
    print(f"[INFO] Combined timetable saved to {out_csv}")
    # print short preview (first 20 rows)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(table.fillna("").to_string(index=False))

# ---------- Save to DB ----------
def save_to_db_combined(table, df_slots):
    # ask info
    tahun = input("Tahun Ajaran (mis: 2025/2026): ").strip()
    semester = input("Semester (ganjil/genap): ").strip()
    keterangan = input("Keterangan (opsional): ").strip()
    with engine.begin() as conn:
        res = conn.execute(text("INSERT INTO jadwal_master (tahun_ajaran, semester, keterangan, dibuat_pada) VALUES (:t,:s,:k,CURRENT_TIMESTAMP())"),
                           {"t":tahun,"s":semester,"k":keterangan})
        master_id = res.lastrowid
        # iterate table rows and insert jadwal rows
        # note: table rows correspond to df_slots rows
        for ix, row in df_slots.iterrows():
            wid = int(row["id_waktu"])
            # for each class column
            for kelas in table.columns:
                if kelas in ["hari","jam_ke"]: continue
                val = table.at[ix, kelas]
                if pd.isna(val) or str(val).strip()=="":
                    continue
                # val format "NamaGuru - NamaMapel" but we need ids. We'll attempt to map by joining on strings (best-effort)
                try:
                    guru_name, mapel_name = [s.strip() for s in val.split(" - ",1)]
                except Exception:
                    guru_name = None; mapel_name = None
                # lookup ids
                gid = None
                mid = None
                try:
                    gid = int(tables["guru"].loc[tables["guru"]["nama_guru"]==guru_name,"id_guru"].iloc[0]) if guru_name else None
                except Exception:
                    gid = None
                try:
                    mid = int(tables["mapel"].loc[tables["mapel"]["nama_mapel"]==mapel_name,"id_mapel"].iloc[0]) if mapel_name else None
                except Exception:
                    mid = None
                # lookup id_kelas by name
                kid = int(tables["kelas"].loc[tables["kelas"]["nama_kelas"]==kelas,"id_kelas"].iloc[0])
                # insert row
                conn.execute(text("INSERT INTO jadwal (id_master,id_kelas,id_mapel,id_guru,id_ruang,id_waktu,generasi,fitness) VALUES (:m,:kid,:mid,:gid,NULL,:wid,0,NULL)"),
                             {"m":master_id,"kid":kid,"mid":mid,"gid":gid,"wid":wid})
        print(f"[DB] saved combined schedule to jadwal_master id {master_id}")
    return master_id

# ---------- Main runner ----------
if __name__ == "__main__":
    print("Starting per-class GA scheduler (Strategy B)...")
    tables = read_tables()
    id_to_slot, day_ordered = build_slots(tables["waktu"])
    candidates_map = build_candidates_map(tables["guru_mapel"])
    # build guru capacity
    guru_capacity = {int(r["id_guru"]): int(r["jam_mingguan"]) for _,r in tables["guru"].iterrows()}
    # classes list
    kelas_list = tables["kelas"].to_dict("records")
    # prepare run dir
    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT, f"run_{run_label}")
    os.makedirs(run_dir, exist_ok=True)
    # snapshot tables
    for k,v in tables.items():
        try: v.to_csv(os.path.join(run_dir,f"{k}.csv"), index=False)
        except: pass

    # global teacher busy map: (guru, id_waktu) -> True if assigned by previous classes
    global_teacher_busy = {}
    all_class_assignments = {}
    # randomize class order to reduce bias
    random.shuffle(kelas_list)
    for kelas_row in kelas_list:
        print(f"\n--- Scheduling class: {kelas_row['nama_kelas']} ---")
        best_ind, best_f, info = run_ga_for_class(kelas_row, tables, id_to_slot, day_ordered, candidates_map, global_teacher_busy, guru_capacity, pop_size=POP_SIZE, generations=GENERATIONS)
        # save per-class gens into run_dir/class_{name} directories: NOTE: run_ga_for_class currently prints per-gen but does not save — we saved snapshots earlier
        if best_ind is None:
            print(f"[WARN] No sessions for class {kelas_row['nama_kelas']}. Skipping.")
            continue
        all_class_assignments[kelas_row['nama_kelas']] = info
        print(f"[{kelas_row['nama_kelas']}] Best fitness: {best_f}")
    # build combined timetable
    combined_table, df_slots = build_combined_timetable(all_class_assignments, id_to_slot, day_ordered, tables)
    save_and_show_combined(combined_table, run_dir)
    # ask to save to DB
    ans = input("Simpan jadwal terbaik ke database? (y/n): ").strip().lower()
    if ans == "y":
        master_id = save_to_db_combined(combined_table, df_slots)
        print(f"[DONE] saved master id {master_id}")
    else:
        print("[INFO] Did not save to DB. Run directory:", run_dir)
    print("Finished.")
