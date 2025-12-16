from data_loader import load_data
from session_builder import build_sessions
from initializer import generate_individual,feasibility_check, safe_generate
from fitness import fitness
from exporter import save_generation
from db_insert import insert_jadwal_master, insert_jadwal_detail
import datetime, os
from export_generation_table import export_generation_table
from data_loader import load_master_map


POP = 50
GEN = 100

def main():
    gm, waktu_ok, waktu_block = load_data()
    kelas_map, mapel_map, guru_map = load_master_map()
    sessions = build_sessions(gm)

    waktu_map = {w["id_waktu"]: w for w in waktu_ok}
    run_dir = f"results/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir)

    feasibility_check(sessions, waktu_ok)
    population = [safe_generate(sessions, waktu_ok) for _ in range(POP)]

    best = None
    best_fit = -1

    for g in range(GEN):
        scored = []
        for ind in population:
            f = fitness(ind, sessions, waktu_map)
            scored.append((f, ind))

        scored.sort(reverse=True, key=lambda x: x[0])
        best_gen_fit, best_gen_ind = scored[0]

        print(f"[Gen {g:04d}] best={best_gen_fit}")

        # 🔥 EXPORT TABEL GENERASI
        export_generation_table(
            run_dir=run_dir,
            gen_label=f"{g:04d}",
            individual=best_gen_ind,
            sessions=sessions,
            waktu_map=waktu_map,
            kelas_map=kelas_map,
            mapel_map=mapel_map,
            guru_map=guru_map
        )

        if best_gen_fit > best_fit:
            best_fit = best_gen_fit
            best = best_gen_ind

        population = [x[1] for x in scored[:POP//2]] * 2

    export_generation_table(
        run_dir=run_dir,
        gen_label="BEST",
        individual=best,
        sessions=sessions,
        waktu_map=waktu_map,
        kelas_map=kelas_map,
        mapel_map=mapel_map,
        guru_map=guru_map
    )

    print("\nBest fitness overall:", best_fit)

    if input("Simpan jadwal terbaik ke database? (y/n): ").lower() == "y":
        tahun = input("Tahun Ajaran (mis: 2025/2026): ")
        semester = input("Semester (ganjil/genap): ")

        id_master = insert_jadwal_master(tahun, semester)
        insert_jadwal_detail(id_master, best, sessions)

        print("✔ Jadwal berhasil disimpan ke database")
    else:
        print("Jadwal tidak disimpan")


if __name__ == "__main__":
    main()
