"""
run_scheduler.py

Runner utama:
- Baca tabel
- Bangun sessions dan time slots
- Ambil konfigurasi GA dari tabel konfigurasi_ag jika ada
- Generate population awal (atau load dari file preprocessed)
- Jalankan GA dan tampilkan + simpan jadwal tiap generasi
"""
from data_preprocessor import read_tables, build_sessions, build_time_slots, generate_initial_population
from genetic_algorithm import run_ga
import os

def main():
    print("Membaca tabel dari database...")
    tables = read_tables()

    print("Membangun sessions...")
    sessions = build_sessions(tables)
    print(f"Total sessions: {len(sessions)}")

    print("Membangun time slots...")
    time_slots = build_time_slots(tables)
    print(f"Total time slots: {len(time_slots)}")

    # konfigurasi dari tabel konfigurasi_ag (jika ada)
    konf = tables.get("konfigurasi_ag")
    if konf is not None and not konf.empty:
        row = konf.iloc[0]
        pop_size = int(row.get("ukuran_populasi", 50))
        pc = float(row.get("probabilitas_crossover", 0.8))
        pm = float(row.get("probabilitas_mutasi", 0.1))
        gens = int(row.get("jumlah_generasi", 50))
    else:
        pop_size = 30
        pc = 0.8
        pm = 0.1
        gens = 20

    print(f"GA config -> pop_size={pop_size}, crossover_prob={pc}, mutation_prob={pm}, generations={gens}")

    # generate initial population
    population = generate_initial_population(sessions, time_slots, pop_size=pop_size)

    # create results folder
    os.makedirs("results", exist_ok=True)

    # jalankan GA (akan mencetak jadwal terbaik tiap generasi)
    best_chrom, best_fitness = run_ga(
        population=population,
        sessions=sessions,
        time_slots=time_slots,
        generations=gens,
        pop_size=pop_size,
        crossover_prob=pc,
        mutation_prob=pm,
        save_csv_dir="results",
        save_best_to_db=False  # ubah jadi True jika ingin menyimpan setiap generasi ke tabel jadwal
    )

    print("\n=== GA SELESAI ===")
    print("Fitness terbaik (final):", best_fitness)
    # Simpan hasil akhir ke CSV final
    final_csv = os.path.join("results", f"final_best_fitness_{best_fitness:.4f}.csv")
    # run_ga sudah menyimpan setiap generasi; jika perlu kita simpan final lagi:
    # (but run_ga already saved gen files)
    print("Hasil disimpan di folder 'results'.")

if __name__ == "__main__":
    main()
