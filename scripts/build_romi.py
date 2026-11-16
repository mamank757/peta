#!/usr/bin/env python3
"""Fase MJO (1-8) dari ROMI (NOAA PSL) -> data/waves/romi.json

Simpan sebagai scripts/build_romi.py.
Nama keluaran 'romi.json' sengaja tidak berawalan mjo/kelvin/rossby, supaya tidak
terhapus oleh pembersihan glob di build_waves.py.
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://psl.noaa.gov/mjo/mjoindex/romi.cpcolr.1x.txt"
OUT = Path("data/waves/romi.json")
MAX_AGE_DAYS = 7
N_JEJAK = 40


def ke_rmm(pc1, pc2):
    """Konvensi NOAA PSL: ROMI(PC2) setara RMM1 dan -ROMI(PC1) setara RMM2."""
    return pc2, -pc1


def fase_dari(rmm1, rmm2):
    """Sudut di bidang (RMM1, RMM2): 180-225° = fase 1 ... 135-180° = fase 8."""
    ang = float(np.degrees(np.arctan2(rmm2, rmm1)) % 360)
    return (int(ang // 45) + 4) % 8 + 1, ang


def unduh():
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(URL, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"unduhan ROMI gagal: {last}")


rows = []
for ln in unduh().splitlines():
    p = ln.split()
    if len(p) < 7:
        continue
    try:
        y, m, d, h = (int(float(x)) for x in p[:4])
        r1, r2, amp = (float(x) for x in p[4:7])
    except ValueError:
        continue  # baris header / teks
    if not (1900 < y < 2200) or not all(abs(v) < 50 for v in (r1, r2, amp)):
        continue  # nilai hilang (-999, dst)
    rows.append((y, m, d, r1, r2, amp))

if not rows:
    sys.exit("ROMI: tidak ada baris data terbaca — tidak ada yang ditulis")

df = pd.DataFrame(rows, columns=["y", "m", "d", "r1", "r2", "amp"])
df["t"] = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d))
last = df.iloc[-1]
age = (pd.Timestamp.now().normalize() - last.t).days
if age > MAX_AGE_DAYS:
    sys.exit(f"ROMI terakhir {last.t.date()} ({age} hari lalu): usang, tidak ditulis")

rmm1, rmm2 = ke_rmm(last.r1, last.r2)
fase, ang = fase_dari(rmm1, rmm2)
aktif = bool(last.amp >= 1)

jejak = [
    dict(tanggal=str(r.t.date()), romi1=round(r.r1, 2), romi2=round(r.r2, 2),
         rmm1=round(r.r2, 2), rmm2=round(-r.r1, 2), amplitudo=round(r.amp, 2))
    for r in df.tail(N_JEJAK).itertuples()
]

meta = dict(
    sumber="NOAA PSL · ROMI (Real-time OLR MJO Index)",
    tanggal=str(last.t.date()),
    romi1=round(float(last.r1), 3),
    romi2=round(float(last.r2), 3),
    rmm1=round(float(rmm1), 3),   # = ROMI PC2
    rmm2=round(float(rmm2), 3),   # = -ROMI PC1
    amplitudo=round(float(last.amp), 3),
    sudut=round(ang, 1),
    aktif=aktif,
    fase=fase if aktif else None,
    faseArah=fase,  # sektor sudut, tetap diisi walau amplitudo < 1
    label=(f"MJO aktif, fase {fase}" if aktif
           else "MJO lemah (amplitudo < 1), tidak ada fase"),
    jejak=jejak,
    catatan="ROMI berbasis OLR saja; mirip RMM tetapi bukan RMM resmi BoM.",
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
print(f"[romi] {meta['tanggal']} PC1={last.r1:.2f} PC2={last.r2:.2f} -> RMM1={rmm1:.2f} RMM2={rmm2:.2f} sudut={ang:.0f} amp={meta['amplitudo']} fase={meta['fase']} ({meta['label']})")
