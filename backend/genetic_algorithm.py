# genetic_algorithm.py
import random
import os
import csv
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Set

from data_preprocessor import SessionBlock, format_schedule_readable

# penalty constants (tune as needed)
P_GURU_CONFLICT = 1000     # very heavy
P_SLOT_BLOCKED = 800       # very heavy
P_GURU_NOT_ALLOWED = 1200  # very heavy
P_EMPTY_GAP = 50           # soft
BASE_SCORE = 10000

def fitness_function(chromosome: Dict[int, Tuple[int,int]],
                     blocks: List[SessionBlock],
                     id_to_waktu: Dict[int, Dict[str, Any]],
                     day_to_slots: Dict[str, List[int]],
                     guru_mapel_active_set: Set[Tuple[int,int,int]]) -> float:
    """
    Compute fitness based on constraints:
      - teacher conflict (hard)
      - blocked slots cannot be used (hard)
      - teacher must be allowed for block's mapel/kelas (hard)
      - no gaps in middle day for each class (soft)
    """
    block_map = {b.block_id: b for b in blocks}
    penalties = 0

    # A. teacher-time conflict -> build per-slot teacher assignment (expand blocks to slots)
    teacher_time = defaultdict(list)  # (slot_id, guru) -> list of block_ids
    class_day_slots = defaultdict(lambda: defaultdict(list))  # kelas -> hari -> list of slot order indices

    # helper: build order_map for day to slot index for gap calculation
    order_map = {}
    for day, slots in day_to_slots.items():
        for idx, sid in enumerate(slots):
            order_map[sid] = (day, idx)

    for bid, assign in chromosome.items():
        start_slot, gid = assign
        b = block_map[bid]
        if start_slot is None or gid is None:
            # heavy penalty for unassigned or missing guru
            penalties += P_GURU_NOT_ALLOWED
            continue
        # find day and day slots list
        if start_slot not in id_to_waktu:
            penalties += P_SLOT_BLOCKED
            continue
        day = id_to_waktu[start_slot]["hari"]
        day_slots = day_to_slots[day]
        # find index of start in day_slots
        try:
            idx = day_slots.index(start_slot)
        except ValueError:
            penalties += P_SLOT_BLOCKED
            continue
        # collect block slot ids
        block_slots = day_slots[idx: idx + b.length]
        if len(block_slots) < b.length:
            # block doesn't fit (end of day) -> heavy penalty
            penalties += P_SLOT_BLOCKED * 2
            continue
        # check each slot for blocked keterangan
        for sid in block_slots:
            ket = id_to_waktu[sid].get("keterangan")
            if ket is not None and str(ket).strip() != "":
                penalties += P_SLOT_BLOCKED
            teacher_time[(sid, gid)].append(bid)
            # register class-day slot order for gaps
            class_day_slots[b.id_kelas][day].append(order_map[sid][1])

        # E: validate guru_mapel_active_set
        key = (int(gid), int(b.id_mapel), int(b.id_kelas))
        if key not in guru_mapel_active_set:
            penalties += P_GURU_NOT_ALLOWED

    # count teacher conflicts: same guru assigned to same slot more than once
    # build map (slot -> dict guru->count)
    slot_guru_counts = defaultdict(lambda: defaultdict(int))
    for (sid, gid), bids in teacher_time.items():
        slot_guru_counts[sid][gid] += len(bids)
    for sid, gdict in slot_guru_counts.items():
        for gid, cnt in gdict.items():
            if cnt > 1:
                penalties += P_GURU_CONFLICT * (cnt - 1)

    # C: empty gaps in middle of day for each class
    for kelas, days in class_day_slots.items():
        for day, indices in days.items():
            if not indices:
                continue
            indices_sorted = sorted(set(indices))
            # if there is a gap between min and max that's not filled
            for i in range(len(indices_sorted)-1):
                if indices_sorted[i+1] - indices_sorted[i] > 1:
                    penalties += P_EMPTY_GAP

    # final normalized fitness
    score = max(0, BASE_SCORE - penalties)
    return score / BASE_SCORE

# GA operators
def tournament_selection(pop, fitnesses, k=3):
    idxs = random.sample(range(len(pop)), k)
    best = idxs[0]
    for i in idxs[1:]:
        if fitnesses[i] > fitnesses[best]:
            best = i
    return pop[best].copy()

def crossover(parent1, parent2, prob=0.8):
    if random.random() > prob:
        return parent1.copy()
    keys = list(parent1.keys())
    if len(keys) < 2:
        return parent1.copy()
    point = random.randint(1, len(keys)-1)
    child = {}
    for i, k in enumerate(keys):
        child[k] = parent1[k] if i < point else parent2[k]
    return child

def mutation(chromosome, blocks: List[SessionBlock], id_to_waktu: Dict[int, Dict[str, Any]], day_to_slots: Dict[str, List[int]], mut_prob=0.12):
    # For each block maybe mutate start_slot or guru
    block_map = {b.block_id: b for b in blocks}
    for bid in list(chromosome.keys()):
        if random.random() < mut_prob:
            b = block_map[bid]
            # choose mutation type
            if random.random() < 0.6:
                # mutate start_slot: choose a random valid start across days for this block
                valid_starts = []
                for day, slots in day_to_slots.items():
                    vs = []
                    # find contiguous valid starts for length b.length
                    n = len(slots)
                    for i in range(0, n - b.length + 1):
                        block_slots = slots[i:i+b.length]
                        ok = True
                        for sid in block_slots:
                            ket = id_to_waktu[sid].get("keterangan")
                            if ket is not None and str(ket).strip() != "" or id_to_waktu[sid].get("jam_ke") is None:
                                ok = False
                                break
                        if ok:
                            vs.append(block_slots[0])
                    valid_starts.extend(vs)
                if valid_starts:
                    chromosome[bid] = (random.choice(valid_starts), chromosome[bid][1])
            else:
                # mutate guru among candidates
                if b.guru_candidates:
                    chromosome[bid] = (chromosome[bid][0], random.choice(b.guru_candidates))
    return chromosome

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

    # normalize population size
    if len(population) < pop_size:
        while len(population) < pop_size:
            population.append(random.choice(population).copy())
    population = population[:pop_size]

    for gen in range(1, generations+1):
        fitnesses = [fitness_function(ch, blocks, id_to_waktu, day_to_slots, guru_mapel_active_set) for ch in population]
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        best = population[best_idx]
        best_fit = fitnesses[best_idx]

        # print only fitness
        print(f"Generasi {gen:03d} | Fitness terbaik: {best_fit:.6f}")

        # save CSV readable
        schedule_read = format_schedule_readable(best, blocks, tables)
        # need guru name lookup
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
                    gname = guru_map.get(gid, None)
                    w.writerow([id_kelas, r["nama_kelas"], r["hari"], r["waktu_mulai"], r["waktu_selesai"], r["nama_mapel"], gid, gname, best_fit])

        # evolve population
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = crossover(p1, p2, prob=crossover_prob)
            child = mutation(child, blocks, id_to_waktu, day_to_slots, mut_prob=mutation_prob)
            new_pop.append(child)
        population = new_pop

    # final best
    final_fits = [fitness_function(ch, blocks, id_to_waktu, day_to_slots, guru_mapel_active_set) for ch in population]
    final_best_idx = max(range(len(population)), key=lambda i: final_fits[i])
    return population[final_best_idx], final_fits[final_best_idx], run_folder
