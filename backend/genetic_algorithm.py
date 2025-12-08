import random
import os
import csv
from data_preprocessor import (
    check_conflicts,
    format_schedule,
    SessionItem
)

# =====================================================
# FITNESS
# =====================================================
def fitness_function(chromosome, session_map):
    conflicts = check_conflicts(chromosome, session_map)
    return 1/(1+conflicts)

# =====================================================
# SELECTION
# =====================================================
def tournament_selection(pop, fit, k=3):
    idx=random.sample(range(len(pop)), k)
    best=idx[0]
    for i in idx[1:]:
        if fit[i]>fit[best]:
            best=i
    return pop[best].copy()

# =====================================================
# CROSSOVER
# =====================================================
def crossover(p1,p2,prob=0.8):
    if random.random()>prob:
        return p1.copy()
    keys=list(p1.keys())
    point=random.randint(1,len(keys)-1)
    child={}
    for i,k in enumerate(keys):
        child[k]=p1[k] if i<point else p2[k]
    return child

# =====================================================
# MUTATION
# =====================================================
def mutation(chrom, sessions, slots, prob=0.1):
    for sid in chrom:
        if random.random()<prob:
            s=sessions[sid-1]
            chrom[sid]=(
                random.choice(slots),
                random.choice(s.ruang_candidates),
                random.choice(s.guru_candidates)
            )
    return chrom

# =====================================================
# MAIN GA LOOP
# =====================================================
def run_ga(population, sessions, slots, generations, run_folder):
    os.makedirs(run_folder, exist_ok=True)
    smap={s.id_session:s for s in sessions}

    for gen in range(1, generations+1):

        fit=[fitness_function(ch, smap) for ch in population]

        best_i=max(range(len(population)), key=lambda i:fit[i])
        best=population[best_i]
        best_fit=fit[best_i]

        # PRINT ONLY FITNESS AND BEST SUMMARY
        print(f"Generasi {gen} | Fitness Terbaik: {best_fit:.4f}")

        schedule=format_schedule(best, sessions)

        # Save CSV generasi
        csv_path=f"{run_folder}/gen_{gen:03d}_fit_{best_fit:.4f}.csv"
        with open(csv_path,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow(["id_kelas","nama_kelas","id_waktu","nama_mapel","id_guru","id_ruang","fitness"])
            for k,rows in schedule.items():
                for r in rows:
                    w.writerow([k,r["nama_kelas"],r["id_waktu"],r["nama_mapel"],r["id_guru"],r["id_ruang"],best_fit])

        # NEXT POP
        new_pop=[]
        while len(new_pop)<len(population):
            p1=tournament_selection(population,fit)
            p2=tournament_selection(population,fit)
            c=crossover(p1,p2)
            c=mutation(c,sessions,slots)
            new_pop.append(c)

        population=new_pop

    return best, best_fit
