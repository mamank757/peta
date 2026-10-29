"""
Ambil angin 10 m dari ECMWF IFS (9 km, model yang sama dengan default Windy)
lewat API Open-Meteo, lalu simpan sebagai grid untuk leaflet-velocity.

Dua grid dibuat (dipilih otomatis di peta sesuai zoom):
  - detail   : Sulsel & sekitarnya, rapat (0.125 derajat ~ 14 km)
  - regional : radius luas Indonesia tengah (0.75 derajat ~ 83 km)

Tiap file berisi beberapa "frame" waktu (tiap 3 jam, 36 jam ke depan).
Browser menginterpolasi antar-frame, jadi angin selalu sesuai jam SEKARANG.

ANGGARAN API (gratis, non-komersial): < 10.000 panggilan/hari, 5.000/jam,
600/menit. Setiap titik grid dihitung 1 panggilan. Script ini memakai
~3.400 titik per eksekusi x 2 eksekusi/hari = ~6.800/hari.
"""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

# ───────────────────────── KONFIGURASI ─────────────────────────
TIERS = {
    "detail":   dict(lat_min=-7.75, lat_max=-1.5, lon_min=117.5, lon_max=122.5, step=0.125),
    "regional": dict(lat_min=-12.0, lat_max=6.0,  lon_min=100.0, lon_max=139.0, step=0.75),
}
FRAME_STEP_H = 3      # jarak antar-frame (jam)
N_FRAMES = 13         # 13 frame x 3 jam = 36 jam
CHUNK = 100           # titik per request
PAUSE_S = 13          # jeda antar request -> ~460 titik/menit (< batas 600/menit)
MODEL = "ecmwf_ifs"   # IFS HRES 9 km
ENDPOINTS = ["https://api.open-meteo.com/v1/ecmwf", "https://api.open-meteo.com/v1/forecast"]
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
DAILY_RUNS = 2        # sesuai cron di workflow
FREE_DAILY_LIMIT = 10_000
# ───────────────────────────────────────────────────────────────

_endpoint_idx = 0


def make_axes(t):
    nx = int(round((t["lon_max"] - t["lon_min"]) / t["step"])) + 1
    ny = int(round((t["lat_max"] - t["lat_min"]) / t["step"])) + 1
    assert abs((nx - 1) * t["step"] - (t["lon_max"] - t["lon_min"])) < 1e-6, "lon tidak kelipatan step"
    assert abs((ny - 1) * t["step"] - (t["lat_max"] - t["lat_min"])) < 1e-6, "lat tidak kelipatan step"
    lons = t["lon_min"] + t["step"] * np.arange(nx)
    lats = t["lat_max"] - t["step"] * np.arange(ny)     # utara -> selatan (aturan leaflet-velocity)
    return np.round(lats, 4), np.round(lons, 4)


def api_get(params):
    """GET dengan retry untuk 429/5xx; pindah endpoint bila 400."""
    global _endpoint_idx
    for attempt in range(6):
        url = ENDPOINTS[_endpoint_idx]
        try:
            r = requests.get(url, params=params, timeout=90)
        except requests.RequestException as e:
            print(f"  koneksi gagal ({e}), ulang...")
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 400 and _endpoint_idx + 1 < len(ENDPOINTS):
            print(f"  {url} -> 400 ({r.text[:120]}); coba endpoint lain")
            _endpoint_idx += 1
            continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = 65 if r.status_code == 429 else 15 * (attempt + 1)
            print(f"  HTTP {r.status_code}, tunggu {wait}s...")
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError("API tetap gagal setelah 6 percobaan")


def pick(hourly, name):
    """Nama variabel bisa polos atau bersufiks model."""
    for k in (name, f"{name}_{MODEL}"):
        if k in hourly:
            return hourly[k]
    raise KeyError(f"{name} tidak ada di respons: {list(hourly)[:6]}")


def fetch_points(points):
    """points: [(lat, lon)] -> (times[list str], speed[n_pts, n_t], direction[n_pts, n_t])."""
    spd_all, dir_all, times = [], [], None
    for i in range(0, len(points), CHUNK):
        chunk = points[i:i + CHUNK]
        params = {
            "latitude": ",".join(f"{p[0]:.4f}" for p in chunk),
            "longitude": ",".join(f"{p[1]:.4f}" for p in chunk),
            "hourly": "wind_speed_10m,wind_direction_10m",
            "models": MODEL,
            "wind_speed_unit": "ms",
            "timezone": "GMT",
            "past_days": 1,
            "forecast_days": 3,
        }
        data = api_get(params)
        if isinstance(data, dict):     # 1 lokasi -> objek tunggal
            data = [data]
        if len(data) != len(chunk):
            raise RuntimeError(f"jumlah lokasi respons {len(data)} != {len(chunk)}")
        for d in data:
            h = d["hourly"]
            if times is None:
                times = h["time"]
            elif h["time"] != times:
                raise RuntimeError("sumbu waktu tidak konsisten antar lokasi")
            spd_all.append(pick(h, "wind_speed_10m"))
            dir_all.append(pick(h, "wind_direction_10m"))
        print(f"  {min(i + CHUNK, len(points))}/{len(points)} titik")
        if i + CHUNK < len(points):
            time.sleep(PAUSE_S)
    to_f = lambda a: np.array([[np.nan if x is None else x for x in row] for row in a], dtype=float)
    return times, to_f(spd_all), to_f(dir_all)


def frame_indices(times, now):
    """Index waktu untuk frame tiap 3 jam, mulai dari (jam-3-kelipatan terakhir - 3 jam)."""
    base = now.replace(minute=0, second=0, microsecond=0)
    base -= dt.timedelta(hours=base.hour % FRAME_STEP_H + FRAME_STEP_H)
    want = [base + dt.timedelta(hours=FRAME_STEP_H * k) for k in range(N_FRAMES)]
    pos = {t: i for i, t in enumerate(times)}
    idx = []
    for w in want:
        key = w.strftime("%Y-%m-%dT%H:%M")
        if key not in pos:
            raise RuntimeError(f"waktu {key} tidak ada di respons API")
        idx.append(pos[key])
    return want, idx


def to_uv(speed, direction):
    """Arah meteorologi (asal angin, derajat) + kecepatan -> u (ke timur), v (ke utara)."""
    rad = np.deg2rad(direction)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def build_pack(tier, t, times, speed, direction, now):
    lats, lons = make_axes(t)
    ny, nx = len(lats), len(lons)
    want, idx = frame_indices(times, now)

    spd, drc = speed[:, idx], direction[:, idx]               # [n_pts, n_frames]
    bad = np.isnan(spd) | np.isnan(drc)
    if bad.any():
        print(f"  peringatan: {bad.sum()} nilai kosong -> diisi 0")
        if bad.mean() > 0.05:
            raise RuntimeError("terlalu banyak data kosong (>5%)")
    spd, drc = np.nan_to_num(spd), np.nan_to_num(drc)
    u, v = to_uv(spd, drc)

    # titik diurutkan lat (utara->selatan) lalu lon -> sudah sama dengan urutan leaflet-velocity
    frames_u = [np.round(u[:, k], 1).tolist() for k in range(len(idx))]
    frames_v = [np.round(v[:, k], 1).tolist() for k in range(len(idx))]
    assert all(len(f) == nx * ny for f in frames_u)

    return {
        "model": "ECMWF IFS 9 km (via Open-Meteo)",
        "tier": tier,
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stepHours": FRAME_STEP_H,
        "times": [w.strftime("%Y-%m-%dT%H:%M:%SZ") for w in want],
        "header": {
            "nx": nx, "ny": ny,
            "lo1": float(lons[0]), "lo2": float(lons[-1]),
            "la1": float(lats[0]), "la2": float(lats[-1]),     # la1 = utara
            "dx": t["step"], "dy": t["step"],
        },
        "u": frames_u, "v": frames_v,
    }


def main():
    now = dt.datetime.now(dt.timezone.utc)
    total = 0
    for t in TIERS.values():
        la, lo = make_axes(t)
        total += len(la) * len(lo)
    per_day = total * DAILY_RUNS
    print(f"Total titik per eksekusi: {total} -> ~{per_day} panggilan/hari (batas gratis {FREE_DAILY_LIMIT})")
    if per_day > FREE_DAILY_LIMIT * 0.9:
        print("PERINGATAN: mendekati batas harian API gratis. Kecilkan grid atau kurangi frekuensi.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    for tier, t in TIERS.items():
        print(f"== Grid '{tier}' ==")
        try:
            lats, lons = make_axes(t)
            points = [(float(la), float(lo)) for la in lats for lo in lons]
            times, speed, direction = fetch_points(points)
            pack = build_pack(tier, t, times, speed, direction, now)
            out = OUT_DIR / f"wind-{tier}.json"
            out.write_text(json.dumps(pack, separators=(",", ":")))
            print(f"  tersimpan {out.name}: {pack['header']['nx']}x{pack['header']['ny']}, "
                  f"{len(pack['times'])} frame, {out.stat().st_size/1024:.0f} KB")
        except Exception as e:      # file lama dibiarkan utuh bila gagal
            failed = True
            print(f"  GAGAL grid '{tier}': {e}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
