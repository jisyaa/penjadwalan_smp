# genetic_algorithm.py
import random
import os
import csv
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Set

from data_preprocessor import SessionBlock, format_schedule_readable

# penalties (tweakable)
P_GURU_CONFLICT = 2000
P_SLOT_BLOCKED = 1500
P_GURU_NOT_ALLOWED = 2000
P_UNASSIGNED_BLOCK = 1500
P_GAP_MISSING = 2000   # VERY HEAVY — user required all teaching slots filled
P_SAME_DAY_SPLIT = 1200

BASE_SCORE = 10000

def fitness_function(chromosome: Dict[int, Tuple[int,int]],
                     blocks: List[SessionBlock],
                     id_to_waktu: Dict[int, Dict[str, Any]],
                     day_to_slots: Dict[str, List[int]],
                     guru_mapel_active_set: Set[Tuple[int,int,int]]) -> float:
    block_map = {b.block_id: b for b in blocks}
    penalties = 0

    # helper: mapping slot->(day,index) for gap & ordering
    day_index = {}
    for day, slots in day_to_slots.items():
        for idx, sid in enumerate(slots):
            day_index[sid] = (day, idx)

    # expand assignments: for each block, compute its occupied slots
    slot_assignments = defaultdict(list)  # slot_id -> list of (block_id, guru)
    class_day_filled = defaultdict(lambda: defaultdict(set))  # class -> day -> set(slot indices)
    block_day_choice = {}  # block_id -> day (to enforce split blocks in different days)

    for bid, assign in chromosome.items():
        start_slot, gid = assign
        b = block_map[bid]
        if start_slot is None or gid is None:
            penalties += P_UNASSIGNED_BLOCK
            continue
        if start_slot not in id_to_waktu:
            penalties += P_SLOT_BLOCKED
            continue
        day = id_to_waktu[start_slot]["hari"]
        # find day slots list
        if day not in day_to_slots:
            penalties += P_SLOT_BLOCKED
            continue
        day_slots = day_to_slots[day]
        try:
            idx = day_slots.index(start_slot)
        except ValueError:
            penalties += P_SLOT_BLOCKED
            continue
        block_slots = day_slots[idx: idx + b.length]
        if len(block_slots) < b.length:
            penalties += P_SLOT_BLOCKED * 2
            continue
        # check none of slots have keterangan or jam_ke is None
        for sid in block_slots:
            info = id_to_waktu[sid]
            if info["keterangan"] is not None and str(info["keterangan"]).strip() != "":
                penalties += P_SLOT_BLOCKED
            if info["jam_ke"] is None:
                penalties += P_SLOT_BLOCKED
            slot_assignments[sid].append((bid, gid))
            # register filled index for class/day
            class_day_filled[b.id_kelas][day].add(day_slots.index(sid))
        # check guru_mapel_active_set
        key = (int(gid), int(b.id_mapel), int(b.id_kelas))
        if key not in guru_mapel_active_set:
            penalties += P_GURU_NOT_ALLOWED
        # record day used by block (for split-block check)
        block_day_choice[bid] = day

    # teacher conflicts: same teacher on same slot in multiple classes
    for sid, lst in slot_assignments.items():
        # count per guru
        per_guru = defaultdict(int)
        for bid, gid in lst:
            per_guru[gid] += 1
        for gid, cnt in per_guru.items():
            if cnt > 1:
                penalties += P_GURU_CONFLICT * (cnt - 1)

    # NO-GAP FULL-DAY: for each class & each day, ALL teaching slots (jam_ke != null and keterangan empty) must be filled
    # compute required teaching slot indices per day (based on day_to_slots & id_to_waktu)
    for kelas, days in class_day_filled.items():
        for day, filled_indices in days.items():
            # collect required indices for that day
            required = []
            slots = day_to_slots[day]
            for idx, sid in enumerate(slots):
                info = id_to_waktu[sid]
                if info["jam_ke"] is None:
                    continue
                if info["keterangan"] is not None and str(info["keterangan"]).strip() != "":
                    continue
                required.append(idx)
            # now check difference between required and filled
            missing = set(required) - set(filled_indices)
            if missing:
                penalties += P_GAP_MISSING * len(missing)

    # Additionally, a class might have zero filled slots for a day but required slots exist: that is missing as well
    # check all classes/day combinations
    # build list of all classes
    all_classes = set([b.id_kelas for b in blocks])
    for kelas in all_classes:
        for day, slots in day_to_slots.items():
            # determine required indices
            required = []
            for idx, sid in enumerate(slots):
                info = id_to_waktu[sid]
                if info["jam_ke"] is None:
                    continue
                if info["keterangan"] is not None and str(info["keterangan"]).strip() != "":
                    continue
                required.append(idx)
            if not required:
                continue
            filled = class_day_filled.get(kelas, {}).get(day, set())
            missing = set(required) - set(filled)
            if missing:
                penalties += P_GAP_MISSING * len(missing)

    # Ensure split-blocks (blocks that originated from same mapel+kls and where that mapel had >3)
    # We can't directly tell which block pairs came from same original mapel - but we can detect same mapel+kls repeated blocks:
    # For each mapel+kls with multiple blocks, ensure not on same day.
    map_blocks = defaultdict(list)
    for b in blocks:
        map_blocks[(b.id_kelas, b.id_mapel)].append(b.block_id)
    for (kelas, mapel), bids in map_blocks.items():
        if len(bids) <= 1:
            continue
        # if there are multiple blocks, their chosen days must be distinct
        days_used = []
        for bid in bids:
            day = block_day_choice.get(bid)
            if day:
                days_used.append(day)
        if len(days_used) != len(set(days_used)):
            penalties += P_SAME_DAY_SPLIT * (len(days_used) - len(set(days_used)))

    score = max(0, BASE_SCORE - penalties)
    return score / BASE_SCORE


# GA ops
def tournament_selection(pop, fitnesses, k=3):
    idxs = random.sample(range(len(pop)), k)
    best = idxs[0]
    for i in idxs[1:]:
        if fitnesses[i] > fitnesses[best]:
            best = i
    return pop[best].copy()

def crossover(p1, p2, prob=0.8):
    if random.random() > prob:
        return p1.copy()
    keys = list(p1.keys())
    if len(keys) < 2:
        return p1.copy()
    pt = random.randint(1, len(keys)-1)
    child = {}
    for i,k in enumerate(keys):
        child[k] = p1[k] if i < pt else p2[k]
    return child

def mutation(chrom, blocks: List[SessionBlock], id_to_waktu: Dict[int, Dict[str, Any]], day_to_slots: Dict[str, List[int]], mut_prob=0.12):
    block_map = {b.block_id: b for b in blocks}
    for bid in list(chrom.keys()):
        if random.random() < mut_prob:
            b = block_map[bid]
            # mutate start_slot or guru
            if random.random() < 0.6:
                # find all valid starts
                valids = []
                for day, slots in day_to_slots.items():
                    # find contiguous valid
                    n = len(slots)
                    for i in range(0, n - b.length + 1):
                        block_slots = slots[i:i+b.length]
                        ok = True
                        for sid in block_slots:
                            info = id_to_waktu[sid]
                            if info["keterangan"] is not None and str(info["keterangan"]).strip() != "":
                                ok = False
                                break
                            if info["jam_ke"] is None:
                                ok = False
                                break
                        if ok:
                            valids.append(block_slots[0])
                if valids:
                    chrom[bid] = (random.choice(valids), chrom[bid][1])
            else:
                if b.guru_candidates:
                    chrom[bid] = (chrom[bid][0], random.choice(b.guru_candidates))
    return chrom

def run_ga(population: List[Dict[int, Tuple[int,int]]],
           blocks: List[SessionBlock],
           waktu_df,
           id_to_waktu: Dict[int, Dict[str, Any]],
           day_to_slots: Dict[str, List[int]],
           guru_mapel_active_set: Set[Tuple[int,int,int]],
           tables: Dict[str, Any],
           generations: int = 30,
           pop_size: int = 30,
           crossover_prob: float = 0.8,
           mutation_prob: float = 0.12,
           save_csv_dir: str = "results",
           run_name: str = None):
    os.makedirs(save_csv_dir, exist_ok=True)
    if run_name is None:
        run_name = f"run_{len(os.listdir(save_csv_dir))+1:03d}"
    run_folder = os.path.join(save_csv_dir, run_name)
    os.makedirs(run_folder, exist_ok=True)

    if len(population) < pop_size:
        while len(population) < pop_size:
            population.append(random.choice(population).copy())
    population = population[:pop_size]

    for gen in range(1, generations+1):
        fitnesses = [fitness_function(ch, blocks, id_to_waktu, day_to_slots, guru_mapel_active_set) for ch in population]
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        best = population[best_idx]
        best_fit = fitnesses[best_idx]

        print(f"Generasi {gen:03d} | Fitness terbaik: {best_fit:.6f}")

        # save CSV readable
        schedule_read = format_schedule_readable(best, blocks, tables)
        # guru lookup
        guru_map = {}
        if "guru" in tables and not tables["guru"].empty:
            for _, r in tables["guru"].iterrows():
                gid = int(r["id_guru"])
                name = r.get("nama_guru") if "nama_guru" in r else r.get("nama")
                guru_map[gid] = name
        csv_path = os.path.join(run_folder, f"gen_{gen:03d}_fit_{best_fit:.6f}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id_kelas","nama_kelas","hari","mulai","selesai","nama_mapel","id_guru","nama_guru","fitness"])
            for id_kelas, rows in schedule_read.items():
                for r in rows:
                    gid = r.get("id_guru")
                    gname = guru_map.get(gid, "")
                    w.writerow([id_kelas, r["nama_kelas"], r["hari"], r["waktu_mulai"], r["waktu_selesai"], r["nama_mapel"], gid, gname, best_fit])

        # evolve
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = crossover(p1, p2, prob=crossover_prob)
            child = mutation(child, blocks, id_to_waktu, day_to_slots, mut_prob=mutation_prob)
            new_pop.append(child)
        population = new_pop

    final_fits = [fitness_function(ch, blocks, id_to_waktu, day_to_slots, guru_mapel_active_set) for ch in population]
    best_idx = max(range(len(population)), key=lambda i: final_fits[i])
    return population[best_idx], final_fits[best_idx], run_folder
