"""
genetic_algorithm.py

Implementasi GA:
- fitness_function (berdasarkan konflik)
- tournament_selection
- crossover
- mutation
- run_ga (loop penuh)
"""
import random
import os
import csv
from typing import List, Dict, Tuple, Any

from data_preprocessor import (
    check_conflicts,
    format_schedule,
    save_schedule,
    SessionItem
)

# ============================================================
# FITNESS FUNCTION
# ============================================================
def fitness_function(chromosome: Dict[int, Tuple[int, int, int]],
                     session_map: Dict[int, SessionItem]) -> float:
    conf = check_conflicts(chromosome, session_map)
    total_conf = conf["total_conflicts"]
    return 1.0 / (1.0 + total_conf)


# ============================================================
# SELECTION — TOURNAMENT
# ============================================================
def tournament_selection(population, fitnesses, k=3):
    idx = random.sample(range(len(population)), k)
    best = idx[0]
    for i in idx[1:]:
        if fitnesses[i] > fitnesses[best]:
            best = i
    return population[best].copy()


# ============================================================
# CROSSOVER — ONE POINT
# ============================================================
def crossover(parent1, parent2, prob=0.8):
    if random.random() > prob:
        return parent1.copy()

    keys = list(parent1.keys())
    if len(keys) < 2:
        return parent1.copy()

    point = random.randint(1, len(keys) - 1)

    child = {}
    for i, sid in enumerate(keys):
        if i < point:
            child[sid] = parent1[sid]
        else:
            child[sid] = parent2[sid]

    return child


# ============================================================
# MUTATION
# ============================================================
def mutation(chromosome, sessions, time_slots, mut_prob=0.1):
    for sid in chromosome.keys():
        if random.random() < mut_prob:
            s = sessions[sid - 1]  # aman karena id_session dibuat berurutan
            chromosome[sid] = (
                random.choice(time_slots),
                random.choice(s.ruang_candidates) if s.ruang_candidates else None,
                random.choice(s.guru_candidates) if s.guru_candidates else None
            )
    return chromosome


# ============================================================
# GA LOOP — WITH SCHEDULE PRINT EVERY GENERATION
# ============================================================
def run_ga(
    population: List[Dict[int, Tuple[int, int, int]]],
    sessions: List[SessionItem],
    time_slots: List[int],
    generations: int = 50,
    pop_size: int = 50,
    crossover_prob: float = 0.8,
    mutation_prob: float = 0.1,
    save_csv_dir: str = "results",
    save_best_to_db: bool = False
):
    os.makedirs(save_csv_dir, exist_ok=True)
    session_map = {s.id_session: s for s in sessions}

    # Pastikan ukuran populasi sama dengan pop_size
    if len(population) != pop_size:
        while len(population) < pop_size:
            population.append(random.choice(population).copy())
        population = population[:pop_size]

    # ================= LOOP GENERASI =================
    for gen in range(1, generations + 1):

        fitnesses = [
            fitness_function(ch, session_map)
            for ch in population
        ]

        # Best individu generasi ini
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        best = population[best_idx]
        best_fit = fitnesses[best_idx]

        # Info konflik
        conf = check_conflicts(best, session_map)

        # ---- PRINT RINGKASAN ----
        print("\n" + "=" * 60)
        print(f" GENERASI {gen} — FITNESS: {best_fit:.4f} — KONFLIK: {conf['total_conflicts']}")
        print("=" * 60)

        # ---- CETAK JADWAL PER KELAS ----
        schedule = format_schedule(best, sessions)
        for id_kelas, rows in schedule.items():
            print(f"\nKELAS: {rows[0]['nama_kelas']}   (ID: {id_kelas})")
            print("id_waktu | mapel                     | guru | ruang")
            print("-" * 55)
            for r in rows:
                print(f"{r['id_waktu']:>8} | {r['nama_mapel'][:25]:25} | {str(r['id_guru']):4} | {str(r['id_ruang']):4}")

        # ---- SIMPAN CSV GENERASI INI ----
        csv_path = os.path.join(save_csv_dir, f"gen_{gen:03d}_fit_{best_fit:.4f}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["id_kelas", "nama_kelas", "id_waktu", "nama_mapel", "id_guru", "id_ruang", "fitness"])
            for id_kelas, rows in schedule.items():
                for r in rows:
                    wr.writerow([
                        id_kelas, r["nama_kelas"], r["id_waktu"],
                        r["nama_mapel"], r["id_guru"], r["id_ruang"], best_fit
                    ])
        print(f"[✓] CSV disimpan: {csv_path}")

        # ---- SIMPAN KE DB (opsional) ----
        if save_best_to_db:
            save_schedule(best, sessions, generation=gen, fitness=best_fit)

        # ---- GENERATE GENERASI BARU ----
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = crossover(p1, p2, prob=crossover_prob)
            child = mutation(child, sessions, time_slots, mut_prob=mutation_prob)
            new_pop.append(child)

        population = new_pop

    # ==== RETURN HASIL TERAKHIR ====
    final_fitnesses = [fitness_function(ch, session_map) for ch in population]
    best_idx = max(range(len(population)), key=lambda i: final_fitnesses[i])
    return population[best_idx], final_fitnesses[best_idx]
