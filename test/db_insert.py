from db import get_connection
from datetime import datetime

def insert_jadwal_master(tahun_ajaran, semester):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO jadwal_master (tahun_ajaran, semester, tanggal_generate)
        VALUES (%s, %s, %s)
    """, (tahun_ajaran, semester, datetime.now()))

    conn.commit()
    id_master = cur.lastrowid
    conn.close()

    return id_master

def insert_jadwal_detail(id_master, best_schedule, sessions):
    conn = get_connection()
    cur = conn.cursor()

    # mapping sid → session
    session_map = {s["sid"]: s for s in sessions}

    for sid, slot_ids in best_schedule.items():
        s = session_map[sid]

        for id_waktu in slot_ids:
            cur.execute("""
                INSERT INTO jadwal 
                (id_master, id_kelas, id_mapel, id_guru, id_waktu)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                id_master,
                s["id_kelas"],
                s["id_mapel"],
                s["id_guru"],
                id_waktu
            ))

    conn.commit()
    conn.close()
