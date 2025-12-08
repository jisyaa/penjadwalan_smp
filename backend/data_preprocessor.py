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

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True
)

# =======================================================
# DATA CLASS
# =======================================================
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

# =======================================================
# LOAD TABLES
# =======================================================
def read_tables():
    tables = {}
    with engine.connect() as conn:
        for t in ["guru", "mapel", "kelas", "ruang", "waktu", "guru_mapel"]:
            df = pd.read_sql(text(f"SELECT * FROM {t}"), conn)
            tables[t] = df
    return tables

# =======================================================
# BUILD SESSIONS
# =======================================================
def build_sessions(tables):
    guru_mapel = tables["guru_mapel"]
    mapel = tables["mapel"].set_index("id_mapel")
    kelas = tables["kelas"].set_index("id_kelas")
    ruang = tables["ruang"]

    sessions = []
    sid = 1

    for _, row in guru_mapel.iterrows():
        if row["aktif"] != "aktif":
            continue

        id_guru = int(row["id_guru"])
        id_mapel = int(row["id_mapel"])
        id_kelas = int(row["id_kelas"])

        jam = int(mapel.loc[id_mapel, "jam_per_minggu"])
        nama_mapel = mapel.loc[id_mapel, "nama_mapel"]
        nama_kelas = kelas.loc[id_kelas, "nama_kelas"]

        kategori = str(mapel.loc[id_mapel].get("kategori", "")).lower()

        ruang_tetap = kelas.loc[id_kelas].get("id_ruang")
        ruang_candidates = []

        if kategori in ("lab", "praktek", "laboratorium"):
            ruang_candidates = ruang[ruang["tipe"].str.contains("lab", case=False, na=False)]["id_ruang"].tolist()
        else:
            ruang_candidates = [int(ruang_tetap)] if ruang_tetap else ruang[ruang["tipe"]=="kelas"]["id_ruang"].tolist()

        guru_candidates = guru_mapel[
            (guru_mapel["id_mapel"] == id_mapel) & 
            (guru_mapel["id_kelas"] == id_kelas) &
            (guru_mapel["aktif"]=="aktif")
        ]["id_guru"].tolist()

        if not guru_candidates:
            guru_candidates = [id_guru]

        for _ in range(jam):
            sessions.append(SessionItem(
                id_session=sid,
                id_kelas=id_kelas,
                nama_kelas=nama_kelas,
                id_mapel=id_mapel,
                nama_mapel=nama_mapel,
                guru_candidates=guru_candidates.copy(),
                ruang_candidates=ruang_candidates.copy()
            ))
            sid += 1

    return sessions

# =======================================================
# TIME SLOTS
# =======================================================
def build_time_slots(tables):
    waktu = tables["waktu"]
    invalid = ["Istirahat", "Ishoma", "Ekstrakulikuler", "Upacara"]
    waktu = waktu[~waktu["keterangan"].isin(invalid)]
    return waktu["id_waktu"].tolist()

# =======================================================
# INITIAL POPULATION
# =======================================================
def generate_initial_population(sessions, time_slots, pop_size=30):
    pop = []
    for _ in range(pop_size):
        chrom = {}
        for s in sessions:
            chrom[s.id_session] = (
                random.choice(time_slots),
                random.choice(s.ruang_candidates),
                random.choice(s.guru_candidates),
            )
        pop.append(chrom)
    return pop

# =======================================================
# CONFLICT CHECKER
# =======================================================
def check_conflicts(chromosome, session_map):
    conflicts = 0
    teacher = defaultdict(list)
    room = defaultdict(list)
    kelas = defaultdict(list)

    for sid, (w, r, g) in chromosome.items():
        session = session_map[sid]
        teacher[(g, w)].append(sid)
        room[(r, w)].append(sid)
        kelas[(session.id_kelas, w)].append(sid)

    for _, v in teacher.items():
        if len(v)>1: conflicts+=len(v)-1
    for _, v in room.items():
        if len(v)>1: conflicts+=len(v)-1
    for _, v in kelas.items():
        if len(v)>1: conflicts+=len(v)-1

    return conflicts

# =======================================================
# FORMAT SCHEDULE
# =======================================================
def format_schedule(chromosome, sessions):
    smap = {s.id_session:s for s in sessions}
    result = defaultdict(list)
    for sid,(w,r,g) in chromosome.items():
        s=smap[sid]
        result[s.id_kelas].append({
            "id_waktu": w,
            "nama_mapel": s.nama_mapel,
            "nama_kelas": s.nama_kelas,
            "id_ruang": r,
            "id_guru": g
        })
    for k in result:
        result[k]=sorted(result[k], key=lambda x:x["id_waktu"])
    return result

# =======================================================
# SAVE TO DB
# =======================================================
def save_schedule_to_db(best, sessions, year, semester):
    smap = {s.id_session: s for s in sessions}
    rows = []

    # 1. simpan ke jadwal_master
    with engine.begin() as conn:
        res = conn.execute(
            text("""
                INSERT INTO jadwal_master (tahun_ajaran, semester, keterangan)
                VALUES (:tahun_ajaran, :semester, :ket)
            """),
            {
                "tahun_ajaran": year,
                "semester": semester,
                "ket": f"Hasil GA tahun {year} semester {semester}"
            }
        )
        id_master = res.lastrowid

    # 2. siapkan data jadwal
    for sid, (w, r, g) in best.items():
        s = smap[sid]
        rows.append({
            "id_master": id_master,
            "id_kelas": s.id_kelas,
            "id_mapel": s.id_mapel,
            "id_guru": g,
            "id_ruang": r,
            "id_waktu": w,
            "generasi": 0,
            "fitness": 0
        })

    df = pd.DataFrame(rows)

    # 3. insert ke tabel jadwal
    with engine.begin() as conn:
        df.to_sql("jadwal", conn, if_exists="append", index=False)

    print(f"✔ Jadwal berhasil disimpan ke database dengan id_master = {id_master}")
