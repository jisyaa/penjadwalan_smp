from collections import defaultdict
import random

# ============================================================
# CEK DASAR
# ============================================================
def feasibility_check(sessions, waktu_tersedia):
    slot_per_day = defaultdict(int)
    for w in waktu_tersedia:
        slot_per_day[w["hari"]] += 1

    max_slot = max(slot_per_day.values())

    for s in sessions:
        if s["durasi"] > max_slot:
            raise Exception(
                f"TIDAK FEASIBLE: durasi={s['durasi']} "
                f"slot per hari max={max_slot}"
            )


# ============================================================
# CONSTRUCTIVE INITIALIZER (CLASS-BASED)
# ============================================================
def generate_individual(sessions, waktu_tersedia):
    schedule = {}                      # sid -> [id_waktu]
    kelas_slot = defaultdict(set)      # id_kelas -> set(id_waktu)

    # group session per kelas
    sessions_by_kelas = defaultdict(list)
    for s in sessions:
        sessions_by_kelas[s["id_kelas"]].append(s)

    # group waktu per hari
    waktu_by_day = defaultdict(list)
    for w in waktu_tersedia:
        waktu_by_day[w["hari"]].append(w)

    for hari in waktu_by_day:
        waktu_by_day[hari].sort(key=lambda x: x["jam_ke"])

    hari_order = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"]

    kelas_list = list(sessions_by_kelas.keys())
    random.shuffle(kelas_list)

    for id_kelas in kelas_list:

        sess = sorted(
            sessions_by_kelas[id_kelas],
            key=lambda x: x["durasi"],
            reverse=True
        )

        for s in sess:
            placed = False

            for hari in hari_order:
                if hari not in waktu_by_day:
                    continue

                slots = waktu_by_day[hari]

                for i in range(len(slots)):
                    block = slots[i:i + s["durasi"]]
                    if len(block) != s["durasi"]:
                        continue

                    # cek bentrok kelas SAJA
                    if any(b["id_waktu"] in kelas_slot[id_kelas] for b in block):
                        continue

                    # tempatkan
                    schedule[s["sid"]] = []
                    for b in block:
                        schedule[s["sid"]].append(b["id_waktu"])
                        kelas_slot[id_kelas].add(b["id_waktu"])

                    placed = True
                    break

                if placed:
                    break

            if not placed:
                raise Exception(
                    f"Gagal tempatkan kelas={id_kelas} "
                    f"mapel={s['id_mapel']} durasi={s['durasi']}"
                )

    return schedule


# ============================================================
# SAFE GENERATE
# ============================================================
def safe_generate(sessions, waktu_ok, max_try=100):
    for _ in range(max_try):
        try:
            return generate_individual(sessions, waktu_ok)
        except:
            continue
    raise Exception("Tidak bisa membuat individu valid")
