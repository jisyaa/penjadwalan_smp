from db import get_connection

def load_data():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT gm.id, gm.id_guru, gm.id_mapel, gm.id_kelas,
               m.jam_per_minggu
        FROM guru_mapel gm
        JOIN mapel m ON gm.id_mapel = m.id_mapel
        WHERE gm.aktif = 'aktif'
    """)
    guru_mapel = cur.fetchall()

    cur.execute("""
        SELECT * FROM waktu
        WHERE keterangan IS NULL OR keterangan = ''
        ORDER BY hari, jam_ke
    """)
    waktu_tersedia = cur.fetchall()

    cur.execute("""
        SELECT * FROM waktu
        WHERE keterangan IS NOT NULL AND keterangan <> ''
    """)
    waktu_terlarang = cur.fetchall()

    conn.close()
    return guru_mapel, waktu_tersedia, waktu_terlarang

def load_master_map():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT id_kelas, nama_kelas FROM kelas")
    kelas_map = {r["id_kelas"]: r["nama_kelas"] for r in cur.fetchall()}

    cur.execute("SELECT id_mapel, nama_mapel FROM mapel")
    mapel_map = {r["id_mapel"]: r["nama_mapel"] for r in cur.fetchall()}

    cur.execute("SELECT id_guru, nama_guru FROM guru")
    guru_map = {r["id_guru"]: r["nama_guru"] for r in cur.fetchall()}

    conn.close()
    return kelas_map, mapel_map, guru_map

