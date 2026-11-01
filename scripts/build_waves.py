#!/usr/bin/env python3
"""
Peta gelombang ekuator (MJO, Kelvin, Rossby) dari data NOAA.

Sumber (semua NOAA PSL THREDDS/OPeNDAP):
  OLR harian NOAA CDR v2 (1°, digrid ke 2,5°): Datasets/olr_cdr_day/olr.day.mean.v2.nc (+ klimatologi 1991-2020)
  Angin 850 hPa harian (NCEP/NCAR R1): Datasets/ncep.reanalysis/Dailies/pressure/{u,v}wnd.YYYY.nc

Metode: anomali thd klimatologi (3 harmonik) -> filter ruang-waktu Wheeler & Kiladis (1999)
  MJO   : timur,  k=1-5,  periode 30-96 hari
  Kelvin: timur,  k=1-14, periode 2,5-30 hari, kedalaman ekuivalen 8-90 m
  ER n=1: barat,  k=1-10, periode 9-72 hari,   kedalaman ekuivalen 8-90 m

Keluaran per gelombang di data/waves/: {w}.png, {w}.json, {w}-vec.json, {w}-hov.png
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from scipy.ndimage import zoom

PSL = "https://psl.noaa.gov/thredds/dodsC/Datasets/"
OLR_URL = PSL + "olr_cdr_day/olr.day.mean.v2.nc"          # NOAA OLR CDR v2, 1°, jeda ±2 hari
OLR_LTM = PSL + "olr_cdr_day/olr.day.ltm.v2.1991-2020.nc"
WND_URL = PSL + "ncep.reanalysis/Dailies/pressure/{v}.{y}.nc"
WND_LTM = PSL + "ncep.reanalysis.derived/pressure/{v}.day.ltm.1991-2020.nc"

NDAYS = 1500                      # ~4 tahun: untuk filter + sigma referensi
MAX_AGE_DAYS = 7                  # tolak data lebih tua dari ini
LAT_T = np.arange(-30, 30.1, 2.5)
LON_T = np.arange(0, 360, 2.5)
LAT = (-30, 30)                   # domain filter
OUT_LAT, OUT_LON = (-25, 25), (30, 210)   # domain tampilan
BOX = dict(lat=(-10, 5), lon=(95, 141))   # kotak Indonesia (indeks regional)
OUT = Path("data/waves")

A, G, OMEGA = 6.371e6, 9.8, 7.292e-5
BETA = 2 * OMEGA / A

WAVES = ("mjo", "kelvin", "rossby")
VMAX = dict(mjo=30, kelvin=15, rossby=15)                 # W/m2 skala warna
HOV_BAND = dict(mjo=(-10, 10), kelvin=(-10, 10), rossby=(-15, 15))
UPSTREAM = dict(mjo=(60, 95), kelvin=(60, 95), rossby=(141, 176))  # kotak hulu
LAGS = dict(mjo=range(4, 15), kelvin=range(1, 7), rossby=range(2, 11))
SRC = "NOAA PSL · OLR terinterpolasi + NCEP/NCAR R1 · filter Wheeler-Kiladis"


# --------------------------------------------------------------- filter
def band_mask(nt, nx, kind):
    """Mask (nt, nx) di ruang (frekuensi, bilangan gelombang). Timur: s>0 & f>0."""
    f = -np.fft.fftfreq(nt)[:, None]               # siklus/hari (tanda dibalik: timur => f*s>0)
    s = np.fft.fftfreq(nx, 1.0 / nx)[None, :]
    af, as_ = np.abs(f), np.abs(s)
    east, west = (f * s) > 0, (f * s) < 0

    def kel(h):                                     # garis dispersi Kelvin, siklus/hari
        return np.sqrt(G * h) * as_ / (2 * np.pi * A) * 86400

    def er(h):                                      # garis dispersi ER n=1, siklus/hari
        k, c = as_ / A, np.sqrt(G * h)
        return BETA * k / (k ** 2 + 3 * BETA / c) / (2 * np.pi) * 86400

    if kind == "mjo":
        m = east & (as_ >= 1) & (as_ <= 5) & (af >= 1 / 96) & (af <= 1 / 30)
    elif kind == "kelvin":
        m = (east & (as_ >= 1) & (as_ <= 14) & (af >= 1 / 30) & (af <= 1 / 2.5)
             & (af >= kel(8)) & (af <= kel(90)))
    else:
        m = (west & (as_ >= 1) & (as_ <= 10) & (af >= 1 / 72) & (af <= 1 / 9)
             & (af >= er(8)) & (af <= er(90)))
    return m.astype(float)


def wk_filter(x, kind, ntaper=30, npad=30):
    """x: (waktu, lintang, bujur global). Ujung akhir diperpanjang meluruh ke nol
    (mengurangi kebocoran spektral); 3-5 hari terakhir tetap kurang andal."""
    x = np.nan_to_num(x).astype(np.float64)
    nt = x.shape[0]
    w = np.ones(nt)
    w[:ntaper] = 0.5 * (1 - np.cos(np.pi * np.arange(ntaper) / ntaper))
    x = x * w[:, None, None]
    ramp = 0.5 * (1 + np.cos(np.pi * np.arange(1, npad + 1) / npad))
    xp = np.concatenate([x, x[-1][None] * ramp[:, None, None]], 0)
    m = band_mask(xp.shape[0], xp.shape[2], kind)
    F = np.fft.ifft2(np.fft.fft2(xp, axes=(0, 2)) * m[:, None, :], axes=(0, 2)).real
    return F[:nt]


# --------------------------------------------------------------- data
def retry(fn, n=4):
    for i in range(n):
        try:
            return fn()
        except Exception as e:                      # unduhan OPeNDAP terpotong dll.
            print(f"percobaan {i + 1}/{n} gagal: {e}", file=sys.stderr)
            time.sleep(5 * (i + 1))
    raise RuntimeError("unduhan gagal setelah beberapa percobaan")


def first_var(ds):
    for k, v in ds.data_vars.items():
        if {"time", "lat", "lon"} <= set(v.dims):
            return k
    raise KeyError(f"tidak ada variabel (time,lat,lon): {list(ds.data_vars)}")


def to_grid(da):
    """Ke grid 2,5° (sama dengan angin NCEP): interpolasi linear separabel, bujur melingkar,
    lintang dijepit (ekstrapolasi konstan di tepi). Tanpa xarray.interp."""
    if da.sizes["lat"] == len(LAT_T) and da.sizes["lon"] == len(LON_T):
        return da
    da = da.transpose("time", "lat", "lon")
    lat, lon = da.lat.values.astype(float), da.lon.values.astype(float)
    arr = np.concatenate([da.values[..., -1:], da.values, da.values[..., :1]], axis=-1)
    lon_e = np.concatenate([[lon[-1] - 360], lon, [lon[0] + 360]])
    j = np.clip(np.searchsorted(lon_e, LON_T, side="right") - 1, 0, len(lon_e) - 2)
    wj = (LON_T - lon_e[j]) / (lon_e[j + 1] - lon_e[j])
    arr = arr[..., j] * (1 - wj) + arr[..., j + 1] * wj
    i = np.clip(np.searchsorted(lat, LAT_T, side="right") - 1, 0, len(lat) - 2)
    wi = np.clip((LAT_T - lat[i]) / (lat[i + 1] - lat[i]), 0, 1)[:, None]
    arr = arr[:, i, :] * (1 - wi) + arr[:, i + 1, :] * wi
    return xr.DataArray(arr.astype(np.float32), dims=("time", "lat", "lon"),
                        coords={"time": da["time"], "lat": LAT_T, "lon": LON_T}, attrs=da.attrs)


def valid_sub(piece, lo, hi, tries=3):
    """Muat potongan; bila ada nilai di luar [lo,hi] (unduhan rusak), muat ulang."""
    for k in range(tries):
        out = retry(lambda: sub(piece))
        bad = int(np.nansum((out.values < lo) | (out.values > hi)))
        if bad == 0:
            return out
        print(f"  potongan berisi {bad} nilai di luar rentang — unduh ulang ({k + 1}/{tries})", file=sys.stderr)
    return out


def sub(da):
    """Potong lintang (urutan asli bisa menurun), muat ke memori, urutkan naik."""
    lat = da.lat.values
    sl = slice(LAT[1], LAT[0]) if lat[0] > lat[-1] else slice(*LAT)
    return da.sel(lat=sl).load().sortby("lat")


def clim_smooth(ltm):
    f = np.fft.rfft(ltm, axis=0)
    f[4:] = 0                                       # pertahankan 3 harmonik pertama
    return np.fft.irfft(f, n=ltm.shape[0], axis=0)


def anomaly(da, ltm):
    c = clim_smooth(ltm.values)
    doy = np.minimum(da.time.dt.dayofyear.values, c.shape[0]) - 1
    return da.values - c[doy]


def fix_time(da):
    da = da.copy()
    da["time"] = pd.to_datetime(da.time.values).floor("D")
    return da


def load_olr():
    ds = xr.open_dataset(OLR_URL)
    var = first_var(ds)
    da = ds[var]
    n = da.sizes["time"]
    starts = list(range(max(0, n - NDAYS), n, 250))
    parts = [valid_sub(da.isel(time=slice(a, min(a + 250, n))), 50, 400) for a in starts]
    da = xr.concat(parts, "time")
    raw = da.values
    out = int(np.nansum((raw < 50) | (raw > 400)))
    print(f"variabel={var} satuan={da.attrs.get('units')} min={np.nanmin(raw):.1f} max={np.nanmax(raw):.1f} "
          f"NaN_asli={np.isnan(raw).mean():.1%} di_luar_50-400={out}")
    da = da.where((da >= 50) & (da <= 400))            # nilai isian/ngawur -> NaN
    da = fix_time(to_grid(da))
    da = da.interpolate_na(dim="time", limit=5)         # celah pendek: interpolasi linear
    dl = xr.open_dataset(OLR_LTM, decode_times=False)
    lv = first_var(dl)
    ltm = valid_sub(dl[lv], 50, 400)
    ltm = to_grid(ltm.where((ltm >= 50) & (ltm <= 400)))
    lv_ = ltm.values
    print(f"klimatologi: variabel={lv} langkah={ltm.shape[0]} min={np.nanmin(lv_):.1f} max={np.nanmax(lv_):.1f} NaN={np.isnan(lv_).mean():.1%}")
    anom = anomaly(da, ltm)
    t = pd.to_datetime(da.time.values)
    print(f"OLR: {t[0].date()} .. {t[-1].date()} grid={da.shape} NaN_akhir={np.isnan(da.values).mean():.1%} "
          f"hari_kosong_penuh={int(np.isnan(da.values).all(axis=(1, 2)).sum())} std_anom={np.nanstd(anom):.1f} W/m2")
    for y in sorted(set(t.year)):
        m = t.year == y
        print(f"  {y}: NaN={np.isnan(da.values[m]).mean():.1%} std_anom={np.nanstd(anom[m]):.1f} "
              f"p1/p50/p99={np.nanpercentile(anom[m], [1, 50, 99]).round(0).tolist()}")
    sd = float(np.nanstd(anom))
    if not (np.isfinite(sd) and 3 < sd < 60) or not (100 < np.nanmin(lv_) and np.nanmax(lv_) < 400):
        sys.exit(f"DATA TIDAK MASUK AKAL: std_anom={sd:.1f}, klimatologi {np.nanmin(lv_):.1f}..{np.nanmax(lv_):.1f} — tidak ada yang ditulis")
    la_std = np.nanstd(anom, axis=(0, 2))
    print("  std_anom per lintang (-30,-15,0,15,30):", [round(float(la_std[i]), 1) for i in (0, 6, 12, 18, 24)])
    return anom, da.time.values, da.lat.values, da.lon.values


def load_wind(var, t0, t1):
    parts = []
    for y in range(pd.Timestamp(t0).year, pd.Timestamp(t1).year + 1):
        ds = xr.open_dataset(WND_URL.format(v=var, y=y))[var].sel(level=850)
        parts.append(valid_sub(ds, -100, 100))
    da = xr.concat(parts, "time")
    da = fix_time(da.where(abs(da) < 100)).sel(time=slice(t0, None))
    dl = xr.open_dataset(WND_LTM.format(v=var), decode_times=False)[var].sel(level=850)
    ltm = valid_sub(dl, -100, 100)
    ltm = ltm.where(abs(ltm) < 100)
    return anomaly(da, ltm), da.time.values


# --------------------------------------------------------------- produk
def box_mean(F, lat, lon, lat_rng, lon_rng):
    la = (lat >= lat_rng[0]) & (lat <= lat_rng[1])
    lo = (lon >= lon_rng[0]) & (lon <= lon_rng[1])
    w = np.cos(np.deg2rad(lat[la]))[None, :, None]
    return (F[:, la][:, :, lo] * w).sum((1, 2)) / (w.sum() * lo.sum())


def lag_corr(up, ds, lags, n=90):
    a, b = up[-n:], ds[-n:]
    return float(max(np.corrcoef(a[:-L], b[L:])[0, 1] for L in lags))


def edge_skill(anom_rows, wave, lat_r, lon, ref, sigma, n_lo=180, n_hi=60, step=4):
    """Hindcast waktu-nyata: potong deret di T, bandingkan estimasi hari-T dengan nilai akhir."""
    nt = anom_rows.shape[0]
    est, tru = [], []
    for T in range(nt - n_lo, nt - n_hi, step):
        Ft = wk_filter(anom_rows[:T + 1], wave)
        est.append(box_mean(Ft, lat_r, lon, BOX["lat"], BOX["lon"])[-1])
        tru.append(ref[T])
    est, tru = np.array(est), np.array(tru)
    r = float(np.corrcoef(est, tru)[0, 1])
    return dict(korelasi=round(r, 2), kemiringan=round(float(np.polyfit(tru, est, 1)[0]), 2),
                rmseSigma=round(float(np.sqrt(np.mean((est - tru) ** 2)) / sigma), 2), n=int(len(est)))


STOPS = [(-1, (8, 48, 107)), (-0.5, (66, 146, 198)), (0, (255, 255, 255)),
         (0.5, (251, 106, 74)), (1, (165, 15, 21))]


def to_rgba(field, vmax):
    z = np.clip(field / vmax, -1, 1)
    xs = [s[0] for s in STOPS]
    rgb = np.stack([np.interp(z, xs, [s[1][c] for s in STOPS]) for c in range(3)], -1)
    a = 0.9 * np.clip(np.abs(z) / 0.25, 0, 1) * 255
    return np.dstack([rgb, a]).astype(np.uint8)


def save_png(field, vmax, path, zf, flip=True):
    f = zoom(field, zf, order=3 if np.isscalar(zf) else 1)
    img = to_rgba(f, vmax)
    Image.fromarray(img[::-1] if flip else img, "RGBA").save(path, optimize=True)


def crop(lat, lon):
    return ((lat >= OUT_LAT[0]) & (lat <= OUT_LAT[1]),
            (lon >= OUT_LON[0]) & (lon <= OUT_LON[1]))


def write_vec(path, u, v, lat, lon, date):
    la, lo = crop(lat, lon)
    la_s, lo_s = lat[la], lon[lo]
    h = dict(parameterUnit="m.s-1", parameterCategory=2, lo1=float(lo_s[0]), lo2=float(lo_s[-1]),
             la1=float(la_s[-1]), la2=float(la_s[0]), dx=2.5, dy=2.5,
             nx=int(lo.sum()), ny=int(la.sum()), refTime=date + "T00:00:00Z", forecastTime=0)
    pack = [{"header": {**h, "parameterNumber": 2}, "data": np.round(u[la][:, lo][::-1], 2).ravel().tolist()},
            {"header": {**h, "parameterNumber": 3}, "data": np.round(v[la][:, lo][::-1], 2).ravel().tolist()}]
    path.write_text(json.dumps(pack, separators=(",", ":")))


def vorticity(u, v, lat):
    p = np.deg2rad(lat)[:, None]
    d = np.deg2rad(2.5)
    t1 = np.gradient(v, d, axis=1)
    t2 = np.gradient(u * np.cos(p), d, axis=0)
    return (t1 - t2) / (A * np.cos(p))


def process(wave, olr, wnd):
    anom, times, lat, lon = olr
    for old in OUT.glob(f"{wave}*"):
        old.unlink()                                # jangan sisakan hasil lama
    F = wk_filter(anom, wave)
    date = str(pd.Timestamp(times[-1]).date())
    vmax = VMAX[wave]
    la, lo = crop(lat, lon)

    # peta OLR tersaring (hari terakhir)
    save_png(F[-1][la][:, lo], vmax, OUT / f"{wave}.png", 4)
    # Hovmoller 90 hari
    hb = (lat >= HOV_BAND[wave][0]) & (lat <= HOV_BAND[wave][1])
    w = np.cos(np.deg2rad(lat[hb]))[None, :, None]
    hov = (F[-90:][:, hb] * w).sum(1) / w.sum()
    save_png(hov[:, lo], vmax, OUT / f"{wave}-hov.png", (3, 4), flip=False)

    # indeks regional Indonesia
    ser = box_mean(F, lat, lon, BOX["lat"], BOX["lon"])
    up = box_mean(F, lat, lon, BOX["lat"], UPSTREAM[wave])
    sigma = float(ser[60:-60].std())
    val = float(ser[-1])
    if sigma < 0.5:
        raise RuntimeError(f"[{wave}] sigma={sigma:.3f} W/m2 tidak masuk akal (data rusak/kosong?) — dihentikan")
    z = val / sigma
    rho = lag_corr(up, ser, LAGS[wave])
    if wave == "mjo":
        rawb = box_mean(anom, lat, lon, BOX["lat"], BOX["lon"])
        print(f"  kotak Indonesia, anomali mentah: rata-rata={np.nanmean(rawb):.1f} std={np.nanstd(rawb):.1f} W/m2")
    rows = (lat >= BOX["lat"][0]) & (lat <= BOX["lat"][1])
    skill = edge_skill(anom[:, rows], wave, lat[rows], lon, ser, sigma)
    keandalan = "baik" if skill["korelasi"] >= 0.85 else ("sedang" if skill["korelasi"] >= 0.6 else "rendah")
    print(f"  [{wave}] 7 hari terakhir: {np.round(ser[-7:], 1).tolist()} | uji tepi: {skill} -> {keandalan}")
    fase = "konvektif" if z <= -1 else ("tertekan" if z >= 1 else "netral")
    propagates = rho >= 0.5 if wave != "mjo" else True
    aktif = bool(z <= -1 and propagates and keandalan != "rendah")

    la_s, lo_s = lat[la], lon[lo]
    meta = dict(
        gelombang=wave, tanggal=date, sumber=SRC, satuan="W/m²", vmax=vmax,
        bounds=[[float(la_s[0]) - 1.25, float(lo_s[0]) - 1.25], [float(la_s[-1]) + 1.25, float(lo_s[-1]) + 1.25]],
        nilaiBox=round(val, 2), sigma=round(sigma, 2), z=round(z, 2), fase=fase, aktif=aktif,
        korelasiPropagasi=round(rho, 2), indeksKelvin=round(max(0.0, -z), 2),
        hov=dict(hari=90, lon0=float(lo_s[0]), lon1=float(lo_s[-1]),
                 tanggal0=str(pd.Timestamp(times[-90]).date()), tanggal1=date),
        catatan="3-5 hari terakhir kurang andal (efek tepi filter). Bukan pengganti produk resmi CPC.",
        keandalanTepi=skill, keandalan=keandalan,
        vektor=False, anomaliU850=None, anomaliAngin=None, vortisitas=None,
    )

    if wnd:
        try:
            (ua, ut), (va, vt) = wnd
            n = min(len(ut), len(vt))
            Fu, Fv = wk_filter(ua[-n:], wave), wk_filter(va[-n:], wave)
            wdate = str(pd.Timestamp(ut[-1]).date())
            write_vec(OUT / f"{wave}-vec.json", Fu[-1], Fv[-1], lat, lon, wdate)
            u_box = box_mean(Fu, lat, lon, BOX["lat"], BOX["lon"])[-1]
            v_box = box_mean(Fv, lat, lon, BOX["lat"], BOX["lon"])[-1]
            zeta = vorticity(Fu[-1], Fv[-1], lat) * 1e5
            zb = box_mean(zeta[None], lat, lon, BOX["lat"], BOX["lon"])[0]
            meta.update(vektor=True, tanggalAngin=wdate, anomaliU850=round(float(u_box), 2),
                        anomaliAngin=round(float(v_box), 2), vortisitas=round(float(zb), 3))
        except Exception as e:                      # angin gagal: peta OLR tetap terbit
            print(f"[{wave}] angin dilewati: {e}", file=sys.stderr)

    arah = {"mjo": "Timur (30-96 hari)", "kelvin": "Timur (±15-20 m/s)", "rossby": "Barat (ER n=1)"}[wave]
    meta["arah"] = arah
    meta["label"] = {"konvektif": "Fase konvektif di atas Indonesia", "tertekan": "Fase tertekan (kering)",
                     "netral": "Sinyal lemah/netral"}[fase]
    if keandalan == "rendah":
        meta["label"] += " (estimasi tepi kurang andal)"
    meta["dampak"] = {"konvektif": "Peluang hujan meningkat (OLR tersaring negatif)",
                      "tertekan": "Hujan cenderung berkurang",
                      "netral": "Tidak dominan"}[fase]
    (OUT / f"{wave}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[{wave}] {date} nilai={val:.1f} W/m2 z={z:.2f} r={rho:.2f} {fase}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    olr = load_olr()
    t0, t1 = olr[1][0], olr[1][-1]
    age = (pd.Timestamp.now().normalize() - pd.Timestamp(t1)).days
    if age > MAX_AGE_DAYS:
        sys.exit(f"OLR terakhir {pd.Timestamp(t1).date()} ({age} hari lalu): terlalu usang, tidak ada yang ditulis")
    try:
        wnd = (load_wind("uwnd", t0, t1), load_wind("vwnd", t0, t1))
    except Exception as e:
        print(f"angin NCEP tidak tersedia: {e}", file=sys.stderr)
        wnd = None
    for w in WAVES:
        process(w, olr, wnd)


if __name__ == "__main__":
    main()
