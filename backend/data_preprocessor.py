# data_preprocessor.py
import os
import random
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

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


@dataclass
class SessionBlock:
    block_id: int
    id_kelas: int
    nama_kelas: str
    id_mapel: int
    nama_mapel: str
    length: int  # number of contiguous teaching slots
    guru_candidates: List[int]


# ---------- read tables ----------
def read_tables() -> Dict[str, pd.DataFrame]:
    with engine.connect() as conn:
        tables = {}
        for t in ["guru", "mapel", "kelas", "waktu", "guru_mapel"]:
            try:
                df = pd.read_sql_table(t, conn)
            except Exception:
                df = pd.read_sql(text(f"SELECT * FROM {t}"), conn)
            tables[t] = df
    return tables


# ---------- split jam_per_minggu into 1 or 2 blocks ----------
def split_into_blocks(jam: int) -> List[int]:
    if jam <= 3:
        return [jam]
    first = math.ceil(jam / 2)
    second = jam - first
    return [first, second]


# ---------- build session blocks using only guru_mapel aktif ----------
def build_session_blocks(tables: Dict[str, pd.DataFrame]) -> List[SessionBlock]:
    gm = tables["guru_mapel"]
    mapel_df = tables["mapel"].set_index("id_mapel")
    kelas_df = tables["kelas"].set_index("id_kelas")

    active = gm[gm["aktif"] == "aktif"]
    blocks: List[SessionBlock] = []
    bid = 1

    for _, r in active.iterrows():
        id_guru = int(r["id_guru"])
        id_mapel = int(r["id_mapel"])
        id_kelas = int(r["id_kelas"])

        if id_mapel not in mapel_df.index or id_kelas not in kelas_df.index:
            continue

        jam = int(mapel_df.loc[id_mapel, "jam_per_minggu"])
        nama_mapel = str(mapel_df.loc[id_mapel, "nama_mapel"])
        nama_kelas = str(kelas_df.loc[id_kelas, "nama_kelas"])

        candidates = active[
            (active["id_mapel"] == id_mapel) & (active["id_kelas"] == id_kelas)
        ]["id_guru"].astype(int).tolist()
        if not candidates:
            candidates = [id_guru]

        lengths = split_into_blocks(jam)
        for length in lengths:
            blocks.append(SessionBlock(
                block_id=bid,
                id_kelas=id_kelas,
                nama_kelas=nama_kelas,
                id_mapel=id_mapel,
                nama_mapel=nama_mapel,
                length=length,
                guru_candidates=candidates.copy()
            ))
            bid += 1

    return blocks


# ---------- build waktu helpers ----------
def build_waktu_structs(waktu_df: pd.DataFrame):
    df = waktu_df.copy()
    df["hari"] = df["hari"].astype(str)
    # order day slots by waktu_mulai then id_waktu
    order = df.sort_values(["hari", "waktu_mulai", "id_waktu"])
    id_to_waktu = {}
    day_to_slots = defaultdict(list)
    for _, r in order.iterrows():
        iid = int(r["id_waktu"])
        id_to_waktu[iid] = {
            "hari": r["hari"],
            "jam_ke": r.get("jam_ke"),
            "mulai": r.get("waktu_mulai"),
            "selesai": r.get("waktu_selesai"),
            "keterangan": r.get("keterangan")
        }
        day_to_slots[r["hari"]].append(iid)
    return df, id_to_waktu, dict(day_to_slots)


# ---------- valid starting slots for block (consecutive and no keterangan, jam_ke not null) ----------
def valid_starts_for_block(day_slots: List[int], id_to_waktu: Dict[int, Dict[str, Any]], length: int) -> List[int]:
    res = []
    n = len(day_slots)
    for i in range(0, n - length + 1):
        block = day_slots[i:i + length]
        ok = True
        for sid in block:
            info = id_to_waktu[sid]
            if info["keterangan"] is not None and str(info["keterangan"]).strip() != "":
                ok = False
                break
            if info["jam_ke"] is None:
                ok = False
                break
        if ok:
            res.append(block[0])
    return res


# ---------- generate initial population ----------
def generate_initial_population(blocks: List[SessionBlock],
                                id_to_waktu: Dict[int, Dict[str, Any]],
                                day_to_slots: Dict[str, List[int]],
                                pop_size: int = 30) -> List[Dict[int, Tuple[int, int]]]:
    # precompute valid starts per block (across days)
    block_valid = {}
    for b in blocks:
        starts = []
        for day, slots in day_to_slots.items():
            vs = valid_starts_for_block(slots, id_to_waktu, b.length)
            starts.extend(vs)
        block_valid[b.block_id] = starts

    population = []
    for _ in range(pop_size):
        chrom = {}
        for b in blocks:
            starts = block_valid[b.block_id]
            start = random.choice(starts) if starts else None
            guru = random.choice(b.guru_candidates) if b.guru_candidates else None
            chrom[b.block_id] = (start, guru)
        population.append(chrom)
    return population


# ---------- lookup maps ----------
def build_lookup_maps(tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict[Any, Any]]:
    guru = tables["guru"]
    waktu = tables["waktu"]

    id_to_guru = {}
    if not guru.empty:
        for _, r in guru.iterrows():
            gid = int(r["id_guru"])
            name = r.get("nama_guru") if "nama_guru" in r else r.get("nama")
            id_to_guru[gid] = name if name is not None else str(gid)

    id_to_waktu = {}
    if not waktu.empty:
        for _, r in waktu.iterrows():
            id_to_waktu[int(r["id_waktu"])] = {
                "hari": r["hari"],
                "mulai": r["waktu_mulai"],
                "selesai": r["waktu_selesai"],
                "keterangan": r.get("keterangan"),
                "jam_ke": r.get("jam_ke")
            }

    return {"guru": id_to_guru, "waktu": id_to_waktu}


# ---------- format chromosome to readable schedule ----------
def format_schedule_readable(chromosome: Dict[int, Tuple[int, int]],
                             blocks: List[SessionBlock],
                             tables: Dict[str, pd.DataFrame]) -> Dict[int, List[Dict[str, Any]]]:
    id_to_waktu = build_lookup_maps(tables)["waktu"]
    bid_map = {b.block_id: b for b in blocks}
    schedule = defaultdict(list)
    dfw = tables["waktu"]

    for bid, assign in chromosome.items():
        start_slot, gid = assign
        b = bid_map[bid]
        if start_slot is None:
            schedule[b.id_kelas].append({
                "hari": None, "waktu_mulai": None, "waktu_selesai": None,
                "nama_mapel": b.nama_mapel, "nama_kelas": b.nama_kelas, "id_guru": gid, "nama_guru": None
            })
            continue
        day = id_to_waktu[start_slot]["hari"]
        day_slots = dfw[dfw["hari"] == day].sort_values(["waktu_mulai", "id_waktu"])["id_waktu"].tolist()
        idx = day_slots.index(start_slot)
        block_slots = day_slots[idx: idx + b.length]
        if not block_slots:
            schedule[b.id_kelas].append({
                "hari": day, "waktu_mulai": None, "waktu_selesai": None,
                "nama_mapel": b.nama_mapel, "nama_kelas": b.nama_kelas, "id_guru": gid, "nama_guru": None
            })
            continue
        mulai = id_to_waktu[block_slots[0]]["mulai"]
        akhir = id_to_waktu[block_slots[-1]]["selesai"]
        schedule[b.id_kelas].append({
            "hari": day, "waktu_mulai": mulai, "waktu_selesai": akhir,
            "nama_mapel": b.nama_mapel, "nama_kelas": b.nama_kelas, "id_guru": gid, "nama_guru": None
        })

    for k in schedule:
        schedule[k] = sorted(schedule[k], key=lambda x: (x["hari"] if x["hari"] else "", x["waktu_mulai"] if x["waktu_mulai"] else ""))
    return schedule


# ---------- save schedule to DB ----------
def save_schedule_to_db(chromosome: Dict[int, Tuple[int, int]],
                        blocks: List[SessionBlock],
                        tables: Dict[str, pd.DataFrame],
                        year: str,
                        semester: str,
                        generation: int = 0,
                        fitness: float = 0.0):
    bid_map = {b.block_id: b for b in blocks}
    id_to_waktu = build_lookup_maps(tables)["waktu"]
    dfw = tables["waktu"]

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO jadwal_master (tahun_ajaran, semester, keterangan)
            VALUES (:y, :s, :k)
        """), {"y": year, "s": semester, "k": f"Hasil GA {year} {semester}"})
        id_master = int(conn.execute(text("SELECT LAST_INSERT_ID()")).scalar())

        rows = []
        for bid, (start_slot, gid) in chromosome.items():
            b = bid_map[bid]
            if start_slot is None or gid is None:
                continue
            day = id_to_waktu[start_slot]["hari"]
            day_slots = dfw[dfw["hari"] == day].sort_values(["waktu_mulai", "id_waktu"])["id_waktu"].tolist()
            idx = day_slots.index(start_slot)
            block_slots = day_slots[idx: idx + b.length]
            for slot in block_slots:
                rows.append({
                    "id_master": id_master,
                    "id_kelas": b.id_kelas,
                    "id_mapel": b.id_mapel,
                    "id_guru": gid,
                    "id_ruang": None,
                    "id_waktu": slot,
                    "generasi": generation,
                    "fitness": float(fitness)
                })
        if rows:
            df = pd.DataFrame(rows)
            df.to_sql("jadwal", con=conn, if_exists="append", index=False)
    print(f"✔ Jadwal disimpan ke DB dengan id_master = {id_master}")
