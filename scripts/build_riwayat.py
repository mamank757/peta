#!/usr/bin/env python3
"""
Riwayat harian indeks gelombang (MJO, Kelvin, Rossby) "AS-OF" untuk model Apps Script.

Keluaran: data/waves/riwayat.json
  {"kolom":["tanggal","mjo","kelvin","rossby"], "sigma":{...}, "data":[["1997-01-01",0.12,-0.30,0.85], ...]}

Definisi nilai (per tanggal T):
  est(T) = rata-rata kotak Indonesia dari OLR tersaring Wheeler-Kiladis pada hari TERAKHIR jendela 1500 hari
           yang berakhir di T, memakai HANYA data sampai T (persis yang tersedia saat prakiraan dibuat),
           dengan filter, taper, dan perpanjangan ujung yang sama dengan build_waves.py.
  nilai = est(T) / sigma_gelombang   (sigma = simpangan baku est selama riwayat, tetap & disimpan di berkas)

Mengapa tidak identik dengan zKalibrasi di {w}.json: kalibrasi (kemiringan hindcast) dan sigma hanyalah faktor
skala per gelombang. Model (Model.gs) menstandarkan setiap fitur (rata-rata & simpangan baku), sehingga faktor
skala tidak mengubah hasil kNN. Yang penting: setiap baris dihitung dengan cara yang sama.

Efisiensi: rantai taper -> perpanjangan ujung -> FFT -> mask -> titik terakhir -> rata-rata kotak bersifat LINEAR,
jadi untuk jendela tetap ia setara satu kernel (waktu x bilangan gelombang). Seluruh riwayat dihitung dengan satu
korelasi geser. Kesetaraan dengan wk_filter diuji otomatis pada data nyata sebelum berkas ditulis.

Pemakaian:
  python scripts/build_riwayat.py --backfill [--mulai 1997-01-01]   # bangun ulang seluruh riwayat (sekali)
  python scripts/build_riwayat.py                                    # tambah tanggal baru (harian)

Catatan kejujuran "as-of": klimatologi OLR tetap 1991-2020 (juga dipakai di produksi), jadi untuk tahun <2020
klimatologi itu memuat data masa depan (efek pada sinyal antar-musim sangat kecil). Celah data <=5 hari diisi
interpolasi linear dua sisi.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import build_waves as bw

WAVES = ("mjo", "kelvin", "rossby")
NT = bw.NDAYS                                   # panjang jendela = produksi (1500 hari)
NTAPER, NPAD = 30, 30                           # = default wk_filter
UJUNG = os.environ.get("RIWAYAT_UJUNG", "ramp")  # 'ramp' | 'mirror'
BERKAS = bw.OUT / "riwayat.json"
MULAI_DEFAULT = "1997-01-01"
LAT_UNDUH = (-12, 7)                            # cukup untuk baris -10..5 pada grid 2,5°
POTONG = 120                                    # hari per permintaan OPeNDAP
MAKS_BARU = 400                                 # mode harian: lebih dari ini -> minta --backfill

NX = len(bw.LON_T)
KS = np.r_[1:15, NX - 14:NX]                    # indeks FFT bujur untuk s = +-1..+-14 (mask lain nol)
BARIS = np.where((bw.LAT_T >= bw.BOX["lat"][0]) & (bw.LAT_T <= bw.BOX["lat"][1]))[0]
LAT_BARIS = bw.LAT_T[BARIS]
BUJUR_KOTAK = np.where((bw.LON_T >= bw.BOX["lon"][0]) & (bw.LON_T <= bw.BOX["lon"][1]))[0]


# ----------------------------------------------------------------- kernel linear
def kernel_ujung(wave, nt=NT, ext=UJUNG, ntaper=NTAPER, npad=NPAD):
    """Kernel C[k, s]: est(T) = Re sum_{k,s} C[k,s] * A[T-nt+1+k, s], A = FFT bujur dari deret rata-rata-lintang."""
    N = nt + npad
    m = bw.band_mask(N, NX, wave)
    H = np.fft.ifft(m[:, KS], axis=0)                       # H[d mod N] = (1/N) sum_k m e^{+2 pi i k d / N}
    w = np.ones(nt)
    w[:ntaper] = 0.5 * (1 - np.cos(np.pi * np.arange(ntaper) / ntaper))
    k = np.arange(nt)
    c = w[:, None] * H[nt - 1 - k]
    j = np.arange(1, npad + 1)
    fade = 0.5 * (1 + np.cos(np.pi * j / npad))
    Hn = H[(-j) % N]                                        # sampel perpanjangan berjarak d = -j dari titik terakhir
    if ext == "ramp":
        c[nt - 1] += w[nt - 1] * (fade[:, None] * Hn).sum(0)
    else:                                                   # mirror: sampel ke-(nt-1-j) dicerminkan
        c[nt - 1 - j] += w[nt - 1 - j][:, None] * fade[:, None] * Hn
    E = np.exp(2j * np.pi * np.outer(KS, BUJUR_KOTAK) / NX).sum(1) / (NX * len(BUJUR_KOTAK))
    return c * E[None, :]


def koef_bujur(anom_baris):
    """anom_baris (n, baris, bujur) -> A (n, len(KS)): rata-rata berbobot cos(lintang), lalu FFT bujur."""
    x = np.nan_to_num(anom_baris).astype(np.float64)
    wr = np.cos(np.deg2rad(LAT_BARIS))
    deret = (x * wr[None, :, None]).sum(1) / wr.sum()
    return np.fft.fft(deret, axis=1)[:, KS]


def est_sepanjang(A, C):
    """est untuk setiap T >= nt-1 (indeks A); panjang = n - nt + 1."""
    win = sliding_window_view(A, C.shape[0], axis=0)          # (n-nt+1, s, k)
    return np.einsum("jsk,ks->j", win, C, optimize=False).real


def referensi(anom_baris, wave, ext=UJUNG):
    """Cara produksi: wk_filter pada jendela penuh, ambil hari terakhir."""
    F = bw.wk_filter(anom_baris, wave, ext=ext)
    return float(bw.box_mean(F, LAT_BARIS, bw.LON_T, bw.BOX["lat"], bw.BOX["lon"])[-1])


# ----------------------------------------------------------------- data
def sumbu_waktu():
    return bw._fetch(bw.OLR_URL, lambda ds: (pd.to_datetime(ds["time"].values).floor("D"), 0, 1))


def unduh_olr(waktu, t_awal):
    """OLR harian lintang -12..7 dari t_awal sampai akhir data, potongan POTONG hari (tiap potongan divalidasi)."""
    import xarray as xr
    i0 = int(np.searchsorted(waktu, np.datetime64(t_awal)))
    i1 = len(waktu)
    print(f"unduh OLR {pd.Timestamp(waktu[i0]).date()} .. {pd.Timestamp(waktu[i1 - 1]).date()} ({i1 - i0} hari)")
    bagian = []
    for a in range(i0, i1, POTONG):
        b = min(a + POTONG, i1)

        def kerja(ds, a=a, b=b):
            da = ds[bw.first_var(ds)].isel(time=slice(a, b))
            lat = da.lat.values
            sl = slice(LAT_UNDUH[1], LAT_UNDUH[0]) if lat[0] > lat[-1] else slice(*LAT_UNDUH)
            out = da.sel(lat=sl).load().sortby("lat")
            v = out.values
            return out, int(np.nansum((v < 50) | (v > 400))), v.size
        bagian.append(bw._fetch(bw.OLR_URL, kerja))
        if len(bagian) % 10 == 0:
            print(f"  ... {b - i0}/{i1 - i0}")
    return xr.concat(bagian, "time")


def siapkan_anomali(waktu, t_awal):
    da = unduh_olr(waktu, t_awal)
    da = da.where((da >= 50) & (da <= 400))
    da = bw.fix_time(bw.to_grid(da)).isel(lat=BARIS)
    hari = pd.date_range(pd.Timestamp(da.time.values[0]), pd.Timestamp(da.time.values[-1]), freq="D")
    da = da.reindex(time=hari).interpolate_na(dim="time", limit=5)     # waktu bolong -> NaN -> isi celah pendek
    ltm = bw.fetch_series(bw.OLR_LTM, 50, 400, 100, decode_times=False)
    ltm = bw.to_grid(ltm.where((ltm >= 50) & (ltm <= 400))).isel(lat=BARIS)
    anom = bw.anomaly(da, ltm)
    sd = float(np.nanstd(anom))
    print(f"anomali: {hari[0].date()} .. {hari[-1].date()} bentuk={anom.shape} std={sd:.1f} W/m2 NaN={np.isnan(anom).mean():.2%}")
    if not (np.isfinite(sd) and 3 < sd < 60):
        sys.exit(f"DATA TIDAK MASUK AKAL (std anomali {sd:.1f}) — tidak ada yang ditulis")
    return anom, hari


# ----------------------------------------------------------------- berkas
def tulis(rows, sigma):
    BERKAS.parent.mkdir(parents=True, exist_ok=True)
    kepala = dict(kolom=["tanggal", "mjo", "kelvin", "rossby"], sigma=sigma, ujung=UJUNG, jendela=NT,
                  metode="OLR tersaring Wheeler-Kiladis, estimasi waktu-nyata (hanya data sampai tanggal itu) / sigma tetap",
                  diperbarui=pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%MZ"))
    teks = json.dumps(kepala, ensure_ascii=False, separators=(",", ":"))[:-1]
    isi = ",\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
    BERKAS.write_text(teks + ',"data":[\n' + isi + "\n]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="bangun ulang seluruh riwayat")
    ap.add_argument("--mulai", default=MULAI_DEFAULT)
    a = ap.parse_args()

    lama, sigma = [], None
    if BERKAS.exists() and not a.backfill:
        j = json.loads(BERKAS.read_text())
        lama, sigma = j["data"], j.get("sigma")
        if not lama or not sigma:
            sys.exit("riwayat.json tidak lengkap (tanpa data/sigma): jalankan dengan --backfill")
        mulai = pd.Timestamp(lama[-1][0]) + pd.Timedelta(days=1)
    elif not a.backfill:
        print("::warning::riwayat.json belum ada. Jalankan workflow 'build-riwayat' (backfill) sekali.")
        return
    else:
        mulai = pd.Timestamp(a.mulai)

    waktu = sumbu_waktu()
    akhir = pd.Timestamp(waktu[-1])
    if akhir < mulai:
        print(f"tidak ada tanggal baru (OLR terakhir {akhir.date()})")
        return
    n_baru = (akhir - mulai).days + 1
    if lama and n_baru > MAKS_BARU:
        sys.exit(f"{n_baru} hari tertinggal (> {MAKS_BARU}): jalankan ulang dengan --backfill")
    print(f"menghitung {mulai.date()} .. {akhir.date()} ({n_baru} hari), ujung={UJUNG}")

    anom, hari = siapkan_anomali(waktu, mulai - pd.Timedelta(days=NT - 1))
    if len(hari) < NT:
        sys.exit("data OLR kurang dari satu jendela penuh")
    A = koef_bujur(anom)
    tgl = hari[NT - 1:]
    est = {w: est_sepanjang(A, kernel_ujung(w)) for w in WAVES}

    # uji kesetaraan dengan wk_filter pada data nyata (jendela terakhir): hentikan bila beda
    for w in WAVES:
        ref, kita = referensi(anom[-NT:], w), float(est[w][-1])
        print(f"  uji {w}: wk_filter={ref:.5f}  kernel={kita:.5f}")
        if abs(ref - kita) > 1e-6 * max(1.0, abs(ref)):
            sys.exit(f"KERNEL TIDAK SAMA dengan wk_filter untuk {w} — build_waves.py berubah? tidak ada yang ditulis")

    if sigma is None:                               # backfill: sigma tetap dari seluruh est
        sigma = {w: round(float(np.nanstd(est[w])), 4) for w in WAVES}
    keep = np.where(tgl >= mulai)[0]
    baru = []
    for i in keep:
        z = [round(float(est[w][i]) / sigma[w], 3) for w in WAVES]
        if all(np.isfinite(z)):
            baru.append([str(tgl[i].date())] + z)
    rows = lama + baru
    print(f"sigma={sigma}; baris baru={len(baru)}; total={len(rows)}; terakhir={rows[-1]}")
    tulis(rows, sigma)


if __name__ == "__main__":
    main()
