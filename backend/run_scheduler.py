import os
from datetime import datetime
from data_preprocessor import (
    read_tables, build_sessions, build_time_slots,
    generate_initial_population, format_schedule,
    save_schedule_to_db
)
from genetic_algorithm import run_ga

def main():

    # ====== BACA DATA ======
    tables=read_tables()
    sessions=build_sessions(tables)
    slots=build_time_slots(tables)

    # ====== BUAT FOLDER RUN ======
    os.makedirs("results",exist_ok=True)
    existing=[d for d in os.listdir("results") if d.startswith("run_")]
    run_no=len(existing)+1
    run_folder=f"results/run_{run_no:03d}"
    os.makedirs(run_folder)

    print(f"Folder hasil: {run_folder}")

    # ====== POPULASI AWAL ======
    pop=generate_initial_population(sessions, slots, pop_size=30)

    # ====== JALANKAN GA ======
    best, best_fit = run_ga(pop, sessions, slots, generations=20, run_folder=run_folder)

    # ====== TAMPILKAN JADWAL TERBAIK ======
    print("\n=== JADWAL TERBAIK ===")
    schedule=format_schedule(best, sessions)

    for k,rows in schedule.items():
        print(f"\nKELAS: {rows[0]['nama_kelas']}")
        for r in rows:
            print(f"  Waktu {r['id_waktu']} | {r['nama_mapel']} | Guru {r['id_guru']} | Ruang {r['id_ruang']}")

    # ====== TANYA SIMPAN KE DB ======
    ans=input("\nSimpan jadwal terbaik ke database? (y/n): ").lower()
    if ans=="y":
        year=input("Tahun Ajaran (mis: 2024/2025): ")
        sem=input("Semester (ganjil/genap): ")
        save_schedule_to_db(best, sessions, year, sem)
        print("✔ Jadwal tersimpan ke database.")
    else:
        print("❌ Jadwal tidak disimpan.")

if __name__=="__main__":
    main()
