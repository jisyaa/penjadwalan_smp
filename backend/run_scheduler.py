# run_scheduler.py
import os
from datetime import datetime
from data_preprocessor import (
    read_tables, build_session_blocks, build_waktu_structs,
    generate_initial_population, build_lookup_maps, format_schedule_readable,
    save_schedule_to_db
)
from genetic_algorithm import run_ga

def build_guru_mapel_active_set(tables):
    gm = tables["guru_mapel"]
    active = gm[gm["aktif"] == "aktif"]
    s = set()
    for _, r in active.iterrows():
        s.add((int(r["id_guru"]), int(r["id_mapel"]), int(r["id_kelas"])))
    return s

def main():
    print("Membaca tabel dari database...")
    tables = read_tables()

    print("Membangun session blocks dari guru_mapel aktif...")
    blocks = build_session_blocks(tables)
    print(f"Total session blocks (pertemuan): {len(blocks)}")

    print("Membangun struktur waktu...")
    waktu_df, id_to_waktu, day_to_slots = build_waktu_structs(tables["waktu"])
    print(f"Total waktu slots: {len(id_to_waktu)}")

    guru_mapel_active_set = build_guru_mapel_active_set(tables)

    # prepare initial population
    pop_size = 40
    population = generate_initial_population(blocks, id_to_waktu, day_to_slots, pop_size=pop_size)

    # run GA
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    best_chrom, best_fit, run_folder = run_ga(
        population=population,
        blocks=blocks,
        waktu_df=waktu_df,
        id_to_waktu=id_to_waktu,
        day_to_slots=day_to_slots,
        guru_mapel_active_set=guru_mapel_active_set,
        tables=tables,
        generations=30,
        pop_size=pop_size,
        run_name=run_name,
        save_csv_dir="results"
    )

    # show final readable schedule with guru names
    print("\n=== JADWAL TERBAIK (final) ===")
    schedule_read = format_schedule_readable(best_chrom, blocks, tables)
    # map guru id->name
    guru_map = {}
    if "guru" in tables and not tables["guru"].empty:
        for _, r in tables["guru"].iterrows():
            gid = int(r["id_guru"])
            name = r.get("nama_guru") if "nama_guru" in r else r.get("nama")
            guru_map[gid] = name

    for id_kelas, rows in schedule_read.items():
        print(f"\nKELAS: {rows[0]['nama_kelas']} (id_kelas={id_kelas})")
        for r in rows:
            gname = guru_map.get(r.get("id_guru"), str(r.get("id_guru")))
            print(f"  {r['hari']} {r['waktu_mulai']}-{r['waktu_selesai']} | {r['nama_mapel']} | Guru: {gname}")

    # ask to save
    ans = input("\nSimpan jadwal terbaik ke database? (y/n): ").strip().lower()
    if ans == "y":
        year = input("Tahun Ajaran (contoh: 2025/2026): ").strip()
        sem = input("Semester (ganjil/genap): ").strip().lower()
        save_schedule_to_db(best_chrom, blocks, tables, year, sem, generation=0, fitness=best_fit)
        print("✔ Jadwal tersimpan ke database.")
        print("Folder hasil run:", run_folder)
    else:
        print("Batal simpan. Folder hasil run:", run_folder)


if __name__ == "__main__":
    main()
