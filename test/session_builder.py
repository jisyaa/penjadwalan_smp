def build_sessions(guru_mapel):
    sessions = []
    sid = 1

    for gm in guru_mapel:
        jam = gm["jam_per_minggu"]

        def add_session(durasi):
            nonlocal sid
            sessions.append({
                "sid": sid,
                "id_guru": gm["id_guru"],
                "id_mapel": gm["id_mapel"],
                "id_kelas": gm["id_kelas"],
                "durasi": durasi
            })
            sid += 1

        if jam == 1:
            add_session(1)

        elif jam == 3:
            add_session(3)

        elif jam == 4:
            add_session(2)
            add_session(2)

        elif jam == 5:
            add_session(3)
            add_session(2)

        elif jam == 6:
            add_session(3)
            add_session(3)

        else:
            raise ValueError(f"Jam per minggu tidak valid: {jam}")

    return sessions
