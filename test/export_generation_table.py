import csv
import os
from collections import defaultdict
from db import get_connection

def export_generation_table(
    run_dir,
    gen_label,          # "0000", "0001", "BEST"
    individual,
    sessions,
    waktu_map,
    kelas_map,
    mapel_map,
    guru_map
):
    # pivot[(hari, jam)][kelas] = "Mapel - Guru"
    pivot = defaultdict(dict)

    for s in sessions:
        sid = s["sid"]
        if sid not in individual:
            continue

        isi = f"{mapel_map[s['id_mapel']]} - {guru_map[s['id_guru']]}"

        for id_waktu in individual[sid]:
            w = waktu_map[id_waktu]
            pivot[(w["hari"], w["jam_ke"])][kelas_map[s["id_kelas"]]] = isi

    kelas_list = sorted(kelas_map.values())

    path = os.path.join(run_dir, f"gen_{gen_label}.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Hari", "Jam"] + kelas_list)

        for (hari, jam) in sorted(
            pivot.keys(),
            key=lambda x: (
                ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu"].index(x[0]),
                x[1]
            )
        ):
            row = [hari, jam]
            for k in kelas_list:
                row.append(pivot[(hari, jam)].get(k, ""))
            writer.writerow(row)

    return path
