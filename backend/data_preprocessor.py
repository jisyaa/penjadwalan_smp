"""
data_preprocessor.py

Fungsi:
- Baca data penjadwalan dari MySQL
- Bangun struktur 'sessions' (sesi pelajaran per kelas per mapel)
- Tentukan guru kandidat & ruang kandidat untuk tiap sesi
- Fungsi validasi konflik untuk sebuah chromosome (assignment)
- Generator populasi awal acak (untuk AG)
- Utility format_schedule, export, dan simpan ke DB
"""
import os
import random
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "db_penjadwalan")

CONN_STR = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONN_STR, pool_pre_ping=True)

# ---------- Data classes ----------
@dataclass
class SessionItem:
    id_session: int
    id_kelas: int
    nama_kelas: str
    id_mapel: int
    nama_mapel: str
    jam_ke: int = 1
    guru_candidates: List[int] = None
    ruang_candidates: List[int] = None

# ---------- Helpers: load tables ----------
def read_tables() -> Dict[str, pd.DataFrame]:
    """
    Baca tabel utama dari database dan kembalikan dict DataFrame.
    Tabel yang dibaca: guru, mapel, kelas, ruang, waktu, guru_mapel, konfigurasi_ag (opsional).
    """
    with engine.connect() as conn:
        tables = {}
        for t in ["guru", "mapel", "kelas", "ruang", "waktu", "guru_mapel", "konfigurasi_ag"]:
            try:
                df = pd.read_sql_table(t, conn)
            except Exception:
                try:
                    df = pd.read_sql(text(f"SELECT * FROM {t}"), conn)
                except Exception:
                    df = pd.DataFrame()
            tables[t] = df
    return tables

# ---------- Build sessions ----------
def build_sessions(tables: Dict[str, pd.DataFrame]) -> List[SessionItem]:
    guru_mapel = tables.get("guru_mapel", pd.DataFrame())
    mapel = tables.get("mapel", pd.DataFrame()).set_index("id_mapel") if not tables.get("mapel", pd.DataFrame()).empty else pd.DataFrame()
    kelas = tables.get("kelas", pd.DataFrame()).set_index("id_kelas") if not tables.get("kelas", pd.DataFrame()).empty else pd.DataFrame()
    ruang = tables.get("ruang", pd.DataFrame())

    sessions: List[SessionItem] = []
    sid = 1

    if guru_mapel.empty or mapel.empty or kelas.empty:
        return sessions

    # iterate guru_mapel rows that are aktif
    for _, row in guru_mapel.iterrows():
        if row.get("aktif") != "aktif":
            continue
        id_guru = int(row["id_guru"])
        id_mapel = int(row["id_mapel"])
        id_kelas = int(row["id_kelas"])
        if id_mapel not in mapel.index or id_kelas not in kelas.index:
            continue

        jam = int(mapel.loc[id_mapel, "jam_per_minggu"])
        nama_mapel = str(mapel.loc[id_mapel, "nama_mapel"])
        nama_kelas = str(kelas.loc[id_kelas, "nama_kelas"])

        kategori = str(mapel.loc[id_mapel].get("kategori", "")).lower()
        kelas_row = kelas.loc[id_kelas]
        ruang_tetap = int(kelas_row.get("id_ruang")) if not pd.isna(kelas_row.get("id_ruang")) else None

        # ruang candidates
        ruang_candidates: List[int] = []
        if kategori in ("praktek", "lab", "laboratorium"):
            if not ruang.empty:
                labs = ruang[ruang["tipe"].isin(["laboratorium", "lab"]) | (ruang["tipe"].str.contains("lab", na=False))]
                ruang_candidates = labs["id_ruang"].astype(int).tolist()
            if not ruang_candidates and ruang_tetap:
                ruang_candidates = [ruang_tetap]
        else:
            if ruang_tetap:
                ruang_candidates = [ruang_tetap]
            else:
                if not ruang.empty and "tipe" in ruang.columns:
                    ruang_candidates = ruang[ruang["tipe"] == "kelas"]["id_ruang"].astype(int).tolist()
                else:
                    ruang_candidates = ruang["id_ruang"].astype(int).tolist() if not ruang.empty else []

        # guru candidates for that class & mapel
        candidates = guru_mapel[
            (guru_mapel["id_mapel"] == id_mapel) & (guru_mapel["id_kelas"] == id_kelas) & (guru_mapel["aktif"] == "aktif")
        ]["id_guru"].astype(int).tolist()
        if not candidates:
            candidates = [id_guru]

        # create 'jam' sessions
        for _ in range(jam):
            s = SessionItem(
                id_session=sid,
                id_kelas=id_kelas,
                nama_kelas=nama_kelas,
                id_mapel=id_mapel,
                nama_mapel=nama_mapel,
                jam_ke=1,
                guru_candidates=candidates.copy(),
                ruang_candidates=ruang_candidates.copy()
            )
            sessions.append(s)
            sid += 1

    return sessions

# ---------- Build time slots ----------
def build_time_slots(tables: Dict[str, pd.DataFrame]) -> List[int]:
    waktu = tables.get("waktu", pd.DataFrame())
    if waktu.empty:
        return []
    invalid_keywords = ["Istirahat", "Ishoma", "Ekstrakulikuler", "Upacara", "Muhadharah", "Literasi", "Tahfidz"]
    filtered = waktu[~waktu["keterangan"].isin(invalid_keywords)]
    slots = filtered["id_waktu"].astype(int).tolist()
    return slots

# ---------- Chromosome representation ----------
# chromosome: { session_id: (id_waktu, id_ruang, id_guru) }

# ---------- Conflict checker ----------
def check_conflicts(
    chromosome: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]],
    session_map: Dict[int, SessionItem]
) -> Dict[str, Any]:
    """
    Hitung konflik berdasarkan chromosome dan mapping session_id -> SessionItem.
    Menghasilkan summary counts dan details list.
    """
    conflicts = []
    teacher_time = defaultdict(list)
    room_time = defaultdict(list)
    class_time = defaultdict(list)

    for sid, assignment in chromosome.items():
        id_waktu, id_ruang, id_guru = assignment
        # skip incomplete assignments
        if id_waktu is None:
            continue
        # note: id_guru or id_ruang may be None; still counted appropriately
        teacher_time[(id_guru, id_waktu)].append(sid)
        room_time[(id_ruang, id_waktu)].append(sid)
        s = session_map.get(sid)
        if s:
            class_time[(s.id_kelas, id_waktu)].append(sid)

    t_conflicts = 0
    for (g, w), sids in teacher_time.items():
        if g is None:
            continue
        if len(sids) > 1:
            t_conflicts += 1
            conflicts.append({"jenis": "guru_bentrok", "id_guru": g, "id_waktu": w, "sesi": sids})

    r_conflicts = 0
    for (r, w), sids in room_time.items():
        if r is None:
            continue
        if len(sids) > 1:
            r_conflicts += 1
            conflicts.append({"jenis": "ruang_bentrok", "id_ruang": r, "id_waktu": w, "sesi": sids})

    c_conflicts = 0
    for (k, w), sids in class_time.items():
        if len(sids) > 1:
            c_conflicts += 1
            conflicts.append({"jenis": "kelas_bentrok", "id_kelas": k, "id_waktu": w, "sesi": sids})

    return {
        "total_teacher_conflicts": t_conflicts,
        "total_room_conflicts": r_conflicts,
        "total_class_conflicts": c_conflicts,
        "total_conflicts": t_conflicts + r_conflicts + c_conflicts,
        "details": conflicts
    }

# ---------- Initial population generator ----------
def generate_initial_population(sessions: List[SessionItem],
                                time_slots: List[int],
                                pop_size: int = 50) -> List[Dict[int, Tuple[int, int, int]]]:
    population = []
    if not time_slots:
        raise ValueError("time_slots kosong. Pastikan tabel waktu tersedia dan difilter.")
    for _ in range(pop_size):
        chrom = {}
        for s in sessions:
            id_waktu = random.choice(time_slots)
            id_ruang = random.choice(s.ruang_candidates) if s.ruang_candidates else None
            id_guru = random.choice(s.guru_candidates) if s.guru_candidates else None
            chrom[s.id_session] = (id_waktu, id_ruang, id_guru)
        population.append(chrom)
    return population

# ---------- Save jadwal to DB ----------
def save_schedule(chromosome: Dict[int, Tuple[int, int, int]],
                  sessions: List[SessionItem],
                  generation:int=0,
                  fitness:float=0.0,
                  clear_previous:bool=False):
    sid_map = {s.id_session: s for s in sessions}
    inserts = []
    for sid, assign in chromosome.items():
        id_waktu, id_ruang, id_guru = assign
        s = sid_map[sid]
        inserts.append({
            "id_kelas": s.id_kelas,
            "id_mapel": s.id_mapel,
            "id_guru": id_guru,
            "id_ruang": id_ruang,
            "id_waktu": id_waktu,
            "generasi": generation,
            "fitness": float(fitness)
        })
    df = pd.DataFrame(inserts)
    with engine.begin() as conn:
        if clear_previous:
            conn.execute(text("DELETE FROM jadwal WHERE generasi = :g"), {"g": generation})
        df.to_sql("jadwal", con=conn, if_exists="append", index=False)

# ---------- Utility: session -> dict export ----------
def export_preprocessed(sessions: List[SessionItem], time_slots: List[int], out_json="preprocessed.json"):
    sessions_dict = [asdict(s) for s in sessions]
    payload = {"sessions": sessions_dict, "time_slots": time_slots}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Exported preprocessed data to {out_json}")

def format_schedule(chromosome: Dict[int, Tuple[int, int, int]], sessions: List[SessionItem]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Mengembalikan dict: id_kelas -> list of {id_waktu, nama_mapel, nama_kelas, id_ruang, id_guru}
    Sorted by id_waktu.
    """
    session_map = {s.id_session: s for s in sessions}
    schedule = defaultdict(list)
    for sid, (id_waktu, id_ruang, id_guru) in chromosome.items():
        s = session_map.get(sid)
        if not s:
            continue
        schedule[s.id_kelas].append({
            "id_waktu": id_waktu,
            "nama_mapel": s.nama_mapel,
            "nama_kelas": s.nama_kelas,
            "id_ruang": id_ruang,
            "id_guru": id_guru
        })
    for k in schedule:
        schedule[k] = sorted(schedule[k], key=lambda x: (x["id_waktu"] if x["id_waktu"] is not None else 9999))
    return schedule
