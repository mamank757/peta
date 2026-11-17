#!/usr/bin/env python3
"""Fase MJO (1-8) dari NOAA PSL -> data/waves/romi.json

Sumber utama : CORe VPM (NOAA PSL), indeks multivariat tipe RMM (kecepatan potensial 200 hPa +
               angin 200/850 hPa). Tanda PC sudah disesuaikan NOAA agar cocok dengan RMM.
Cadangan     : ROMI (NOAA PSL, OLR saja), dikonversi ke konvensi RMM
               (RMM1 = ROMI PC2, RMM2 = -ROMI PC1 sesuai keterangan NOAA PSL).
Indeks NOAA lain dapat ditambahkan di daftar SUMBER (nama, url, parser, keterangan).

Simpan sebagai scripts/build_romi.py. Nama keluaran 'romi.json' sengaja tidak
berawalan mjo/kelvin/rossby, supaya tidak terhapus oleh glob di build_waves.py.
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

VPM_URL = "https://psl.noaa.gov/mjo/mjoindex/vpm.1x.CORe.txt"
ROMI_URL = "https://psl.noaa.gov/mjo/mjoindex/romi.cpcolr.1x.txt"
OUT = Path("data/waves/romi.json")
MAX_AGE_DAYS = 7
N_JEJAK = 40
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")


def unduh(url):
    err = None
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/plain,*/*"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"unduhan gagal ({url}: {err})")


def fase_dari(rmm1, rmm2):
    """Sudut di bidang (RMM1, RMM2): 180-225° = fase 1 ... 135-180° = fase 8."""
    ang = float(np.degrees(np.arctan2(rmm2, rmm1)) % 360)
    return (int(ang // 45) + 4) % 8 + 1, ang


def parse_romi(teks):
    """Kolom PSL: tahun bulan hari jam PC1 PC2 amplitudo. ROMI(PC2)=RMM1, -ROMI(PC1)=RMM2."""
    rows = []
    for ln in teks.splitlines():
        p = ln.split()
        if len(p) < 7:
            continue
        try:
            y, m, d = int(p[0]), int(p[1]), int(p[2])
            pc1, pc2 = float(p[4]), float(p[5])
        except ValueError:
            continue
        if not (1900 < y < 2200) or abs(pc1) > 50 or abs(pc2) > 50:
            continue
        rows.append((y, m, d, pc2, -pc1))
    return rows


def parse_vpm(teks):
    """Kolom PSL: tahun bulan hari jam PC1 PC2 amplitudo. Tanda PC sudah sesuai RMM: RMM1=PC1, RMM2=PC2."""
    rows = []
    for ln in teks.splitlines():
        p = ln.split()
        if len(p) < 7:
            continue
        try:
            y, m, d = int(p[0]), int(p[1]), int(p[2])
            pc1, pc2 = float(p[4]), float(p[5])
        except ValueError:
            continue
        if not (1900 < y < 2200) or abs(pc1) > 50 or abs(pc2) > 50:
            continue
        rows.append((y, m, d, pc1, pc2))
    return rows


# (nama indeks, url, parser, keterangan sumber, catatan) — yang pertama segar jadi utama,
# sumber lain yang segar ikut disimpan sebagai pembanding
SUMBER = [
    ("VPM", VPM_URL, parse_vpm, "NOAA PSL · CORe VPM (angin + kecepatan potensial, tipe RMM)",
     "Indeks multivariat NOAA PSL (CORe) yang disesuaikan dengan konvensi RMM; nilai hari terakhir dapat direvisi."),
    ("ROMI", ROMI_URL, parse_romi, "NOAA PSL · ROMI (berbasis OLR saja)",
     "Indeks berbasis OLR saja (NOAA PSL); dapat berbeda satu-dua fase dari indeks RMM yang juga memakai angin."),
]


def ke_df(rows):
    df = pd.DataFrame(rows, columns=["y", "m", "d", "rmm1", "rmm2"])
    df["t"] = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d))
    df["amp"] = np.hypot(df.rmm1, df.rmm2)
    # file PSL kadang memuat tanggal ganda: urutkan, ambil baris terakhir tiap tanggal
    df = df.sort_values("t", kind="stable").drop_duplicates(subset="t", keep="last")
    return df.reset_index(drop=True)


def muat(url, parser):
    teks = unduh(url)
    df = ke_df(parser(teks))
    baris = [ln for ln in teks.splitlines() if ln.strip()]
    if df.empty:
        raise RuntimeError(f"tidak ada baris data terbaca; {len(baris)} baris, 3 terakhir: {baris[-3:]}")
    age = (pd.Timestamp.now().normalize() - df.iloc[-1].t).days
    if age > MAX_AGE_DAYS:
        raise RuntimeError(f"data valid terakhir {df.iloc[-1].t.date()} ({age} hari lalu): usang; "
                           f"3 baris terakhir file mentah: {baris[-3:]}")
    return df


def ringkas(nama, ket, df):
    last = df.iloc[-1]
    fase, ang = fase_dari(last.rmm1, last.rmm2)
    aktif = bool(last.amp >= 1)
    sektor, pos = int(ang // 45), ang % 45          # posisi di dalam sektor 45°
    jarak = min(pos, 45 - pos)                      # derajat ke batas fase terdekat
    tetangga = ((sektor + (1 if pos > 22.5 else -1)) % 8 + 4) % 8 + 1
    return dict(
        indeks=nama, sumber=ket, tanggal=str(last.t.date()),
        rmm1=round(float(last.rmm1), 3), rmm2=round(float(last.rmm2), 3),
        amplitudo=round(float(last.amp), 3), sudut=round(ang, 1),
        aktif=aktif, fase=fase if aktif else None, faseArah=fase,
        jarakBatas=round(jarak, 1), faseTetangga=tetangga,
    )


def main():
    segar, galat = [], {}
    for nama, url, parser, ket, catatan in SUMBER:
        try:
            df = muat(url, parser)
        except Exception as e:  # noqa: BLE001
            galat[nama] = str(e)
            print(f"[fase] {nama} tidak tersedia: {e}", file=sys.stderr)
            continue
        segar.append((nama, ket, catatan, df))
    if not segar:
        sys.exit("Tidak ada indeks fase MJO yang bisa dibaca — tidak ada yang ditulis")

    nama, ket, catatan, df = segar[0]
    meta = ringkas(nama, ket, df)
    meta["label"] = (f"MJO aktif, fase {meta['fase']}" if meta["aktif"]
                     else "MJO lemah (amplitudo < 1), tidak ada fase")
    if meta["aktif"] and meta["jarakBatas"] < 10:
        meta["label"] += f" (dekat batas fase {meta['faseTetangga']})"
    meta["jejak"] = [dict(tanggal=str(r.t.date()), rmm1=round(r.rmm1, 2), rmm2=round(r.rmm2, 2),
                          amplitudo=round(r.amp, 2)) for r in df.tail(N_JEJAK).itertuples()]
    meta["catatan"] = catatan
    if len(segar) > 1:
        n2, k2, _, d2 = segar[1]
        meta["pembanding"] = ringkas(n2, k2, d2)
    if galat:
        meta["galatSumber"] = galat

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[fase] {meta['indeks']} {meta['tanggal']} RMM1={meta['rmm1']:.2f} RMM2={meta['rmm2']:.2f} "
          f"sudut={meta['sudut']:.0f} amp={meta['amplitudo']:.2f} fase={meta['fase']} (batas fase {meta['faseTetangga']}: {meta['jarakBatas']:.0f}° lagi)")
    if "pembanding" in meta:
        pb = meta["pembanding"]
        print(f"[fase] pembanding {pb['indeks']} {pb['tanggal']} RMM1={pb['rmm1']:.2f} RMM2={pb['rmm2']:.2f} "
              f"amp={pb['amplitudo']:.2f} fase={pb['fase']} (faseArah={pb['faseArah']})")


if __name__ == "__main__":
    main()
