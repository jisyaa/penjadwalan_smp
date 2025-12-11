#!/usr/bin/env python3
"""
ga_global_hard_slotsharing.py
Improved Global GA (HARD, slot-sharing model) for SMPN 1 Enam Lingkung.

Key changes vs earlier:
- slot-sharing: ordered_slots is time template (40) reused across classes
- no impossible total_slots check
- improved initial population + local-search repair to increase feasibility
- logging of sample invalid reasons when fitness==0 to debug dataset issues
- outputs: results/run_<ts>/gen_XXX/combined_timetable.csv and final/

Dependencies:
pip install sqlalchemy pymysql pandas python-dotenv tqdm
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

# ---------- CONFIG ----------
DB_USER = os.getenv("DB_USER","root")
DB_PASS = os.getenv("DB_PASS","")
DB_HOST = os.getenv("DB_HOST","127.0.0.1")
DB_PORT = os.getenv("DB_PORT","3306")
DB_NAME = os.getenv("DB_NAME","db_penjadwalan")

RESULTS_ROOT = os.getenv("RESULTS_DIR","./results")
POP_SIZE = int(os.getenv("POP_SIZE","40"))
GENERATIONS = int(os.getenv("GENERATIONS","120"))

CONN_STR = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONN_STR, pool_pre_ping=True)

random.seed(12345)

# ---------- Types ----------
Session = namedtuple("Session", ["sid","id_kelas","nama_kelas","id_mapel","nama_mapel","length","split_group"])

# ---------- DB read ----------
def read_tables():
    with engine.connect() as conn:
        tables = {}
        for t in ["guru","mapel","kelas","guru_mapel","waktu"]:
            tables[t] = pd.read_sql(text(f"SELECT * FROM {t}"), conn)
    return tables

# ---------- slots template ----------
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

# ---------- sessions creation ----------
def build_sessions(tables, target_hours_per_class=40):
    gm = tables["guru_mapel"][tables["guru_mapel"]["aktif"]=="aktif"]
    mapel_df = tables["mapel"]
    kelas_df = tables["kelas"]
    sessions=[]
    sid=1
    split_gid=1
    expected=defaultdict(int)
    pairs=gm.groupby(["id_kelas","id_mapel"]).size().reset_index()[["id_kelas","id_mapel"]]
    for _, r in pairs.iterrows():
        id_kelas=int(r["id_kelas"]); id_mapel=int(r["id_mapel"])
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
        expected[nama_kelas]+=jam
    # enforce target hours
    for k in expected: expected[k]=target_hours_per_class
    return sessions, expected

# ---------- candidates (guru options) ----------
def build_candidates(guru_mapel_df):
    cand=defaultdict(list)
    for _, r in guru_mapel_df[guru_mapel_df["aktif"]=="aktif"].iterrows():
        cand[(int(r["id_kelas"]), int(r["id_mapel"]))].append(int(r["id_guru"]))
    return cand

# ---------- utilities ----------
def find_consecutive(start_wid, length, id_to_slot, day_ordered):
    if start_wid not in id_to_slot: return None
    day = id_to_slot[start_wid]["hari"]
    ordered = day_ordered[day]
    try:
        idx = ordered.index(start_wid)
    except ValueError:
        return None
    block = ordered[idx: idx+length]
    if len(block)!=length: return None
    prev=None
    for wid in block:
        jk = id_to_slot[wid]["jam_ke"]
        if prev is not None and jk != prev+1: return None
        prev=jk
    return block

# ---------- greedy placement used for initial population ----------
def greedy_place_all(sessions, id_to_slot, day_ordered, candidates, class_order):
    slots_map={}
    guru_map={}
    busy = {}  # (guru,wid)->True
    class_assigned = defaultdict(set)
    per_class = defaultdict(list)
    for s in sessions: per_class[s.nama_kelas].append(s)
    for cls in class_order:
        if cls not in per_class: continue
        slist = sorted(per_class[cls], key=lambda x:-x.length)
        for s in slist:
            placed=False
            # prefer days with enough free contiguous slots
            days = ["Senin","Selasa","Rabu","Kamis","Jumat"]
            # shuffle but keep slight preference to natural order
            random.shuffle(days)
            for d in days:
                ordered = day_ordered.get(d,[])
                for i in range(len(ordered)):
                    start = ordered[i]
                    block = find_consecutive(start, s.length, id_to_slot, day_ordered)
                    if not block: continue
                    if any(w in class_assigned[cls] for w in block): continue
                    cands = candidates.get((s.id_kelas, s.id_mapel), [])
                    random.shuffle(cands)
                    for g in cands:
                        conflict=False
                        for wid in block:
                            if busy.get((g,wid), False): conflict=True; break
                        if conflict: continue
                        # assign
                        slots_map[s.sid]=block[0]; guru_map[s.sid]=g
                        for wid in block:
                            busy[(g,wid)]=True
                            class_assigned[cls].add(wid)
                        placed=True; break
                    if placed: break
                if placed: break
            if not placed:
                slots_map[s.sid]=None; guru_map[s.sid]=None
    return slots_map, guru_map

def create_initial_population(sessions, id_to_slot, day_ordered, candidates, pop_size):
    class_names = sorted({s.nama_kelas for s in sessions})
    pop=[]
    for i in range(pop_size):
        order = class_names.copy()
        random.shuffle(order)
        smap,gmap = greedy_place_all(sessions, id_to_slot, day_ordered, candidates, order)
        pop.append((smap,gmap))
    return pop

# ---------- local-search repair (hill-climb) ----------
def local_search_improve(individual, sessions, id_to_slot, day_ordered, candidates, guru_capacity, max_iter=200):
    """
    Try to reduce invalidities by moving problematic sessions to alternative starts/gurus.
    Operates per individual and returns improved individual.
    """
    slots_map, guru_map = deepcopy(individual[0]), deepcopy(individual[1])
    # helper: evaluate quick penalty counts (small)
    def quick_penalty(smap, gmap):
        pen = 0
        # missing/nonconsec/gap/teacher conflict/class conflict/split violation/expected mismatch
        # we compute lightweight penalty counts (not full detailed)
        teacher_time=defaultdict(list); class_time=defaultdict(list); assigned_per_class=defaultdict(set); split_days=defaultdict(set)
        for s in sessions:
            st = smap.get(s.sid); g = gmap.get(s.sid)
            if st is None or g is None:
                pen += 1000; continue
            block = find_consecutive(st, s.length, id_to_slot, day_ordered)
            if not block:
                pen += 1000; continue
            for wid in block:
                teacher_time[(g,wid)].append(s.sid); class_time[(s.nama_kelas,wid)].append(s.sid); assigned_per_class[s.nama_kelas].add(wid)
            if s.split_group is not None:
                split_days[s.split_group].add(id_to_slot[block[0]]["hari"])
        for (g,w), lst in teacher_time.items():
            if len(lst)>1: pen += 5000 * (len(lst)-1)
        for (cls,w), lst in class_time.items():
            if len(lst)>1: pen += 5000 * (len(lst)-1)
        for gid, days in split_days.items():
            if len(days) < 2: pen += 2000
        for cls, wids in assigned_per_class.items():
            per_day=defaultdict(list)
            for wid in wids: per_day[id_to_slot[wid]["hari"]].append(id_to_slot[wid]["jam_ke"])
            for day,jks in per_day.items():
                jks_sorted=sorted(jks)
                for i in range(len(jks_sorted)-1):
                    if jks_sorted[i+1] != jks_sorted[i]+1:
                        pen += 1000
        return pen

    base_pen = quick_penalty(slots_map, guru_map)
    # pick problematic sessions: those missing or non-consec or producing conflicts
    for _ in range(max_iter):
        # random select a session to try to improve, biased to ones unassigned or causing problems
        bad_sids=[]
        for s in sessions:
            st=slots_map.get(s.sid); g=guru_map.get(s.sid)
            if st is None or g is None:
                bad_sids.append(s.sid); continue
            if not find_consecutive(st, s.length, id_to_slot, day_ordered):
                bad_sids.append(s.sid)
        if not bad_sids:
            # include random subset to try to improve minor conflicts
            s = random.choice(sessions)
            cand_sids=[s.sid]
        else:
            cand_sids = random.sample(bad_sids, min(len(bad_sids), 6))
        improved=False
        for sid in cand_sids:
            s = next(x for x in sessions if x.sid==sid)
            # try alternative starts and gurus
            tries=[]
            for d, ordered in day_ordered.items():
                for i in range(len(ordered)):
                    st = ordered[i]
                    block = find_consecutive(st, s.length, id_to_slot, day_ordered)
                    if not block: continue
                    tries.append(st)
            random.shuffle(tries)
            cands = candidates.get((s.id_kelas, s.id_mapel), [])
            if not cands: continue
            for st in tries[:30]:  # limit tries to speed
                for g in random.sample(cands, min(len(cands),5)):
                    old_st = slots_map.get(sid); old_g = guru_map.get(sid)
                    slots_map[sid]=st; guru_map[sid]=g
                    new_pen = quick_penalty(slots_map, guru_map)
                    if new_pen < base_pen:
                        base_pen = new_pen
                        improved=True
                        break
                    else:
                        # revert
                        slots_map[sid]=old_st; guru_map[sid]=old_g
                if improved: break
            if improved: break
        if not improved:
            break
    return (slots_map, guru_map)

# ---------- fitness (hard) with diagnostic ----------
def fitness_hard_diagnostic(chrom, sessions, id_to_slot, day_ordered, expected_per_class, candidates, guru_capacity):
    slots_map, guru_map = chrom
    teacher_time=defaultdict(list); class_time=defaultdict(list); assigned_per_class=defaultdict(set); split_days=defaultdict(set)
    # validate all sessions assigned and consecutive
    for s in sessions:
        st = slots_map.get(s.sid); g = guru_map.get(s.sid)
        if st is None or g is None:
            return 0.0, {"reason":"missing_session","sid":s.sid,"s":s}
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block:
            return 0.0, {"reason":"non_consecutive","sid":s.sid,"s":s}
        for wid in block:
            row = id_to_slot.get(wid)
            if row is None:
                return 0.0, {"reason":"invalid_slot","sid":s.sid}
            if row["keterangan"] is not None and str(row["keterangan"]).strip()!="":
                return 0.0, {"reason":"blocked_slot","sid":s.sid,"wid":wid}
        for wid in block:
            teacher_time[(g,wid)].append(s.sid); class_time[(s.nama_kelas,wid)].append(s.sid); assigned_per_class[s.nama_kelas].add(wid)
        if s.split_group is not None:
            split_days[s.split_group].add(id_to_slot[block[0]]["hari"])
    # teacher clash
    for (g,w), lst in teacher_time.items():
        if len(lst)>1:
            return 0.0, {"reason":"teacher_conflict","guru":g,"wid":w,"count":len(lst)}
    # class clash
    for (cls,w), lst in class_time.items():
        if len(lst)>1:
            return 0.0, {"reason":"class_conflict","kelas":cls,"wid":w,"count":len(lst)}
    # split groups on different days
    for gid, days in split_days.items():
        if len(days) < 2:
            return 0.0, {"reason":"split_violation","split_group":gid}
    # no gaps per class per day
    for cls, wids in assigned_per_class.items():
        per_day=defaultdict(list)
        for wid in wids: per_day[id_to_slot[wid]["hari"]].append(id_to_slot[wid]["jam_ke"])
        for day,jks in per_day.items():
            jks_sorted=sorted(jks)
            for i in range(len(jks_sorted)-1):
                if jks_sorted[i+1] != jks_sorted[i]+1:
                    return 0.0, {"reason":"gap","kelas":cls,"day":day}
    # expected hours per class exact
    for cls, expected in expected_per_class.items():
        assigned = len(assigned_per_class.get(cls,set()))
        if assigned != expected:
            return 0.0, {"reason":"expected_hours_mismatch","kelas":cls,"assigned":assigned,"expected":expected}
    # teacher capacity
    teacher_hours=defaultdict(int)
    for (g,w), lst in teacher_time.items(): teacher_hours[g]+=len(lst)
    for g,h in teacher_hours.items():
        cap = guru_capacity.get(g, 9999)
        if h>cap:
            return 0.0, {"reason":"teacher_overload","guru":g,"hours":h,"cap":cap}
    # if reached here, valid; compute small bonus for balance
    bonus=0.0
    for cls, wids in assigned_per_class.items():
        per_day=defaultdict(int)
        for wid in wids: per_day[id_to_slot[wid]["hari"]]+=1
        vals=list(per_day.values()); 
        if len(vals)>0:
            mean=sum(vals)/len(vals)
            var=sum((v-mean)**2 for v in vals)/len(vals)
            bonus += max(0,1.0 - var/25.0)
    fitness = 1.0 + bonus
    return fitness, {"reason":"valid","bonus":bonus}

# ---------- GA operators ----------
def tournament(pop, fitnesses, k=3):
    idxs = random.sample(range(len(pop)), min(k, len(pop)))
    best = max(idxs, key=lambda i: fitnesses[i])
    return deepcopy(pop[best])

def chunk_crossover(p1, p2, sessions, prob=0.85):
    if random.random() > prob:
        return deepcopy(p1), deepcopy(p2)
    a_slots,a_gurus = deepcopy(p1[0]), deepcopy(p1[1])
    b_slots,b_gurus = deepcopy(p2[0]), deepcopy(p2[1])
    for s in sessions:
        if random.random() < 0.5:
            a_slots[s.sid], b_slots[s.sid] = b_slots.get(s.sid), a_slots.get(s.sid)
            a_gurus[s.sid], b_gurus[s.sid] = b_gurus.get(s.sid), a_gurus.get(s.sid)
    return (a_slots,a_gurus),(b_slots,b_gurus)

def chunk_mutation(ind, sessions, id_to_slot, day_ordered, candidates, prob=0.18):
    slots_map, guru_map = deepcopy(ind[0]), deepcopy(ind[1])
    for s in sessions:
        if random.random() < prob:
            starts=[]
            for d, ordered in day_ordered.items():
                for i in range(len(ordered)):
                    st = ordered[i]
                    if find_consecutive(st, s.length, id_to_slot, day_ordered): starts.append(st)
            if starts:
                slots_map[s.sid] = random.choice(starts)
        if random.random() < prob:
            cands = candidates.get((s.id_kelas, s.id_mapel), [])
            if cands:
                guru_map[s.sid] = random.choice(cands)
    return (slots_map, guru_map)

# ---------- save generation/final ----------
def save_generation(run_dir, gen_idx, population, fitnesses, best_idx, sessions, id_to_slot, ordered_slots, tables):
    gen_dir = os.path.join(run_dir, f"gen_{gen_idx:03d}")
    os.makedirs(gen_dir, exist_ok=True)
    best = population[best_idx]; best_f = fitnesses[best_idx]
    rows=[]
    for wid in ordered_slots:
        r = id_to_slot[wid]; rows.append({"id_waktu":wid,"hari":r["hari"],"jam_ke":r["jam_ke"]})
    df_slots = pd.DataFrame(rows)
    class_names = sorted({s.nama_kelas for s in sessions})
    table = pd.DataFrame(index=range(len(df_slots)), columns=["hari","jam_ke"]+class_names)
    table["hari"]=df_slots["hari"]; table["jam_ke"]=df_slots["jam_ke"]
    table.fillna("", inplace=True)
    guru_df = tables["guru"]
    slots_map,guru_map = best
    for s in sessions:
        st = slots_map.get(s.sid); g = guru_map.get(s.sid)
        if st is None or g is None: continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block: continue
        try: nama_g = guru_df.loc[guru_df["id_guru"]==g,"nama_guru"].iloc[0]
        except: nama_g = str(g)
        entry = f"{nama_g} - {s.nama_mapel}"
        for wid in block:
            idx = df_slots.index[df_slots["id_waktu"]==wid].tolist()
            if not idx: continue
            r = idx[0]
            table.at[r, s.nama_kelas] = entry
    table.to_csv(os.path.join(gen_dir,"combined_timetable.csv"), index=False)
    with open(os.path.join(gen_dir,"fitness.json"), "w", encoding="utf-8") as f:
        json.dump({"best":float(best_f),"avg":float(sum(fitnesses)/len(fitnesses)),"worst":float(min(fitnesses))}, f, indent=2)
    return gen_dir, table

def save_final(run_dir, population, fitnesses, best_idx, sessions, id_to_slot, ordered_slots, tables):
    final_dir = os.path.join(run_dir,"final"); os.makedirs(final_dir, exist_ok=True)
    best = population[best_idx]; slots_map,guru_map = best
    rows=[]
    for wid in ordered_slots:
        r = id_to_slot[wid]; rows.append({"id_waktu":wid,"hari":r["hari"],"jam_ke":r["jam_ke"]})
    df_slots = pd.DataFrame(rows)
    class_names = sorted({s.nama_kelas for s in sessions})
    table = pd.DataFrame(index=range(len(df_slots)), columns=["hari","jam_ke"]+class_names)
    table["hari"]=df_slots["hari"]; table["jam_ke"]=df_slots["jam_ke"]
    table.fillna("", inplace=True)
    guru_df = tables["guru"]
    for s in sessions:
        st = slots_map.get(s.sid); g = guru_map.get(s.sid)
        if st is None or g is None: continue
        block = find_consecutive(st, s.length, id_to_slot, day_ordered)
        if not block: continue
        try: nama_g = guru_df.loc[guru_df["id_guru"]==g,"nama_guru"].iloc[0]
        except: nama_g = str(g)
        entry = f"{nama_g} - {s.nama_mapel}"
        for wid in block:
            idx = df_slots.index[df_slots["id_waktu"]==wid].tolist()
            if not idx: continue
            r = idx[0]
            table.at[r, s.nama_kelas] = entry
    table.to_csv(os.path.join(final_dir,"combined_timetable.csv"), index=False)
    with open(os.path.join(final_dir,"fitness.json"), "w", encoding="utf-8") as f:
        json.dump({"best":float(fitnesses[best_idx]),"avg":float(sum(fitnesses)/len(fitnesses)),"worst":float(min(fitnesses))}, f, indent=2)
    return final_dir, table

# ---------- Main ----------
if __name__ == "__main__":
    print("GA Global (HARD, slot-sharing) starting...")
    tables = read_tables()
    id_to_slot, day_ordered, ordered_slots = build_slots(tables["waktu"])
    sessions, expected_per_class = build_sessions(tables, target_hours_per_class=40)
    candidates = build_candidates(tables["guru_mapel"])
    guru_capacity = {int(r["id_guru"]): int(r["jam_mingguan"]) for _,r in tables["guru"].iterrows()}

    print(f"[info] sessions: {len(sessions)}")
    print(f"[info] slots template (per week): {len(ordered_slots)} (used by all classes)")
    # no fatal check; slot-sharing implies reuse across classes

    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT, f"run_{run_label}")
    os.makedirs(run_dir, exist_ok=True)
    for k,v in tables.items():
        try: v.to_csv(os.path.join(run_dir,f"{k}.csv"), index=False)
        except: pass

    # initial pop
    print("[info] creating initial population...")
    population = create_initial_population(sessions, id_to_slot, day_ordered, candidates, POP_SIZE)
    # improve each with local search to increase feasibility
    population = [ local_search_improve(ind, sessions, id_to_slot, day_ordered, candidates, guru_capacity, max_iter=200) for ind in population ]

    fitnesses=[]
    diagnostics=[]
    for ind in population:
        f,diag = fitness_hard_diagnostic(ind, sessions, id_to_slot, day_ordered, expected_per_class, candidates, guru_capacity)
        fitnesses.append(f); diagnostics.append(diag)
    print(f"[info] init avg={sum(fitnesses)/len(fitnesses):.8f} best={max(fitnesses):.8f}")

    best_overall=None; best_f_overall=-1
    # track sample failure reasons
    failure_reasons = Counter()
    for gen in range(1, GENERATIONS+1):
        avg = sum(fitnesses)/len(fitnesses)
        best_f = max(fitnesses); worst_f = min(fitnesses)
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        print(f"[Gen {gen:03d}] avg={avg:.8f} best={best_f:.8f} worst={worst_f:.8f}")
        # save generation
        gen_dir, _ = save_generation(run_dir, gen, population, fitnesses, best_idx, sessions, id_to_slot, ordered_slots, tables)
        # sample diagnostics when best==0 to know why
        if best_f == 0.0:
            # collect small sample of diag reasons
            for d in diagnostics[:10]:
                if isinstance(d, dict) and d.get("reason"):
                    failure_reasons[d["reason"]] += 1
        if best_f > best_f_overall:
            best_f_overall = best_f; best_overall = deepcopy(population[best_idx])
        # build new population
        new_pop=[deepcopy(population[best_idx])]  # elitism 1
        while len(new_pop) < POP_SIZE:
            p1 = tournament(population, fitnesses, k=3)
            p2 = tournament(population, fitnesses, k=3)
            c1,c2 = chunk_crossover(p1,p2,sessions,prob=0.85)
            c1 = chunk_mutation(c1,sessions,id_to_slot,day_ordered,candidates,prob=0.18)
            c2 = chunk_mutation(c2,sessions,id_to_slot,day_ordered,candidates,prob=0.18)
            # local search on kids to improve feasibility
            c1 = local_search_improve(c1, sessions, id_to_slot, day_ordered, candidates, guru_capacity, max_iter=120)
            c2 = local_search_improve(c2, sessions, id_to_slot, day_ordered, candidates, guru_capacity, max_iter=120)
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE: new_pop.append(c2)
        population = new_pop
        fitnesses=[]; diagnostics=[]
        for ind in population:
            f,diag = fitness_hard_diagnostic(ind, sessions, id_to_slot, day_ordered, expected_per_class, candidates, guru_capacity)
            fitnesses.append(f); diagnostics.append(diag)

    # final
    print("GA finished.")
    final_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    final_dir, final_table = save_final(run_dir, population, fitnesses, final_idx, sessions, id_to_slot, ordered_slots, tables)
    print(f"[INFO] final saved to {final_dir}")
    # show sampled failure reasons if any and best fitness reached
    if best_f_overall <= 0:
        print("[WARN] GA could not find any fully-valid solution. Sample failure reasons (counts):")
        print(dict(failure_reasons))
        print("Check dataset completeness (guru_mapel coverage, available teachers per mapel).")
    else:
        print(f"[OK] best fitness achieved: {best_f_overall:.6f}")
    print(final_table.fillna("").head(60).to_string(index=False))
    ans = input("Simpan final ke DB jadwal_master/jadwal? (y/n): ").strip().lower()
    if ans=="y":
        # reuse save function from previous versions - minimal implementation
        slots_map, guru_map = population[final_idx]
        tahun = input("Tahun ajaran (e.g. 2025/2026): ").strip()
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
