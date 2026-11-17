#!/usr/bin/env python3
"""Fase MJO (1-8) -> data/waves/romi.json

Utama : RMM (Wheeler-Hendon) dari BoM Australia, indeks resmi yang dipakai BMKG.
Cadangan: ROMI (NOAA PSL, OLR saja) bila BoM tidak bisa diunduh.
ROMI selalu ikut disimpan sebagai pembanding bila tersedia.

Simpan sebagai scripts/build_romi.py. Nama keluaran 'romi.json' sengaja tidak
berawalan mjo/kelvin/rossby, supaya tidak terhapus oleh glob di build_waves.py.
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BOM_URLS = [
    "https://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
    "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
]
ROMI_URL = "https://psl.noaa.gov/mjo/mjoindex/romi.cpcolr.1x.txt"
OUT = Path("data/waves/romi.json")
MAX_AGE_DAYS = 7
N_JEJAK = 40
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")


def unduh(urls):
    err = None
    for url in ([urls] if isinstance(urls, str) else urls):
        for _ in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/plain,*/*"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.read().decode("utf-8", "replace")
            except Exception as e:  # noqa: BLE001
                err = f"{url}: {e}"
    raise RuntimeError(f"unduhan gagal ({err})")


def fase_dari(rmm1, rmm2):
    """Sudut di bidang (RMM1, RMM2): 180-225° = fase 1 ... 135-180° = fase 8."""
    ang = float(np.degrees(np.arctan2(rmm2, rmm1)) % 360)
    return (int(ang // 45) + 4) % 8 + 1, ang


def parse_bom(teks):
    """Kolom BoM: tahun bulan hari RMM1 RMM2 fase amplitudo [asal]. Nilai hilang 1.E36 / 999."""
    rows = []
    for ln in teks.splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        try:
            y, m, d = int(p[0]), int(p[1]), int(p[2])
            r1, r2 = float(p[3]), float(p[4])
        except ValueError:
            continue
        if not (1900 < y < 2200) or abs(r1) > 50 or abs(r2) > 50:
            continue
        rows.append((y, m, d, r1, r2))
    return rows


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


def ke_df(rows):
    df = pd.DataFrame(rows, columns=["y", "m", "d", "rmm1", "rmm2"])
    df["t"] = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d))
    df["amp"] = np.hypot(df.rmm1, df.rmm2)
    return df.reset_index(drop=True)


def ringkas(df, indeks):
    last = df.iloc[-1]
    fase, ang = fase_dari(last.rmm1, last.rmm2)
    aktif = bool(last.amp >= 1)
    return dict(
        indeks=indeks, tanggal=str(last.t.date()),
        rmm1=round(float(last.rmm1), 3), rmm2=round(float(last.rmm2), 3),
        amplitudo=round(float(last.amp), 3), sudut=round(ang, 1),
        aktif=aktif, fase=fase if aktif else None, faseArah=fase,
    )


def muat(nama, url, parser):
    df = ke_df(parser(unduh(url)))
    if df.empty:
        raise RuntimeError("tidak ada baris data terbaca")
    age = (pd.Timestamp.now().normalize() - df.iloc[-1].t).days
    if age > MAX_AGE_DAYS:
        raise RuntimeError(f"data terakhir {df.iloc[-1].t.date()} ({age} hari lalu): usang")
    return df


def main():
    hasil, galat = {}, {}
    for nama, url, parser in (("RMM", BOM_URLS, parse_bom), ("ROMI", ROMI_URL, parse_romi)):
        try:
            hasil[nama] = muat(nama, url, parser)
        except Exception as e:  # noqa: BLE001
            galat[nama] = str(e)
            print(f"[fase] {nama} tidak tersedia: {e}", file=sys.stderr)
    if not hasil:
        sys.exit("Tidak ada indeks fase MJO yang bisa dibaca — tidak ada yang ditulis")

    utama = "RMM" if "RMM" in hasil else "ROMI"
    df = hasil[utama]
    meta = ringkas(df, utama)
    meta["sumber"] = ("BoM Australia · RMM (Wheeler-Hendon)" if utama == "RMM"
                      else "NOAA PSL · ROMI (OLR saja, bentuk RMM)")
    meta["label"] = (f"MJO aktif, fase {meta['fase']}" if meta["aktif"]
                     else "MJO lemah (amplitudo < 1), tidak ada fase")
    meta["jejak"] = [dict(tanggal=str(r.t.date()), rmm1=round(r.rmm1, 2), rmm2=round(r.rmm2, 2),
                          amplitudo=round(r.amp, 2)) for r in df.tail(N_JEJAK).itertuples()]
    if utama == "RMM":
        meta["catatan"] = "RMM resmi BoM Australia (indeks yang dipakai BMKG). Nilai hari terakhir dapat direvisi."
    else:
        meta["catatan"] = ("BoM RMM tidak bisa diunduh; memakai ROMI (OLR saja), bisa berbeda "
                           "satu-dua fase dari RMM resmi.")
    if utama == "RMM" and "ROMI" in hasil:
        meta["pembanding"] = ringkas(hasil["ROMI"], "ROMI")
    if galat:
        meta["galatSumber"] = galat

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    pb = meta.get("pembanding")
    print(f"[fase] {utama} {meta['tanggal']} RMM1={meta['rmm1']:.2f} RMM2={meta['rmm2']:.2f} "
          f"sudut={meta['sudut']:.0f} amp={meta['amplitudo']:.2f} fase={meta['fase']} ({meta['label']})")
    if pb:
        print(f"[fase] pembanding ROMI {pb['tanggal']} amp={pb['amplitudo']:.2f} fase={pb['fase']}")


if __name__ == "__main__":
    main()
