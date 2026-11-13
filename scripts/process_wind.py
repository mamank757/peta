"""
Ambil angin 10 m dari ECMWF IFS (9 km, model yang sama dengan default Windy)
lewat API Open-Meteo, lalu simpan sebagai grid untuk leaflet-velocity.

Dua grid dibuat (dipilih otomatis di peta sesuai zoom):
  - detail   : Sulsel & sekitarnya (0.25 derajat ~ 28 km)
  - regional : radius luas Indonesia tengah (1.5 derajat ~ 167 km)

Tiap file berisi beberapa "frame" waktu (tiap 3 jam, 36 jam ke depan).
Browser menginterpolasi antar-frame, jadi angin selalu sesuai jam SEKARANG.

ANGGARAN API (gratis, non-komersial): < 10.000 panggilan/hari, 5.000/jam,
600/menit. Setiap titik grid dihitung 1 panggilan.

Perubahan dibanding versi lama:
  - batas waktu internal (DEADLINE) -> skrip berhenti sendiri dengan pesan jelas
  - output langsung tampil di log GitHub Actions (tanpa buffering)
  - isi respons error dicetak; 429 "hourly"/"daily" tidak diulang percuma
  - preflight 1 titik di awal: gagal cepat kalau kuota/endpoint bermasalah
  - exit 1 hanya jika SEMUA grid gagal (grid yang berhasil tetap di-commit)
"""
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.stdout.reconfigure(line_buffering=True)   # log tampil langsung di Actions

# ───────────────────────── KONFIGURASI ─────────────────────────
TIERS = {
    # Untuk resolusi lebih rapat: detail step=0.125, regional step=0.75
    # (total ~3.400 titik/eksekusi, rawan kena batas kuota di runner GitHub)
    "detail":   dict(lat_min=-7.75, lat_max=-1.5, lon_min=117.5, lon_max=122.5, step=0.25),
    "regional": dict(lat_min=-12.0, lat_max=6.0,  lon_min=100.0, lon_max=139.0, step=1.5),
}
FRAME_STEP_H = 3      # jarak antar-frame (jam)
N_FRAMES = 13         # 13 frame x 3 jam = 36 jam
CHUNK = 100           # titik per request
PAUSE_S = 13          # jeda antar request -> ~460 titik/menit (< batas 600/menit)
MODEL = "ecmwf_ifs"   # IFS HRES 9 km
ENDPOINTS = ["https://api.open-meteo.com/v1/ecmwf", "https://api.open-meteo.com/v1/forecast"]
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
DAILY_RUNS = int(os.environ.get("RUNS_PER_DAY", "2"))   # diisi dari workflow
FREE_DAILY_LIMIT = 10_000
MAX_RUNTIME_S = 20 * 60                                  # menyerah setelah 20 menit
# ───────────────────────────────────────────────────────────────

DEADLINE = time.time() + MAX_RUNTIME_S
_endpoint_idx = 0


class QuotaExceeded(RuntimeError):
    """Batas jam/hari API habis: mengulang tidak ada gunanya."""


def sleep_checked(seconds):
    if time.time() + seconds > DEADLINE:
        raise RuntimeError(f"batas waktu {MAX_RUNTIME_S // 60} menit akan terlewati, berhenti")
    time.sleep(seconds)


def make_axes(t):
    nx = int(round((t["lon_max"] - t["lon_min"]) / t["step"])) + 1
    ny = int(round((t["lat_max"] - t["lat_min"]) / t["step"])) + 1
    assert abs((nx - 1) * t["step"] - (t["lon_max"] - t["lon_min"])) < 1e-6, "lon tidak kelipatan step"
    assert abs((ny - 1) * t["step"] - (t["lat_max"] - t["lat_min"])) < 1e-6, "lat tidak kelipatan step"
    lons = t["lon_min"] + t["step"] * np.arange(nx)
    lats = t["lat_max"] - t["step"] * np.arange(ny)     # utara -> selatan (aturan leaflet-velocity)
    return np.round(lats, 4), np.round(lons, 4)


def build_params(chunk):
    return {
        "latitude": ",".join(f"{p[0]:.4f}" for p in chunk),
        "longitude": ",".join(f"{p[1]:.4f}" for p in chunk),
        "hourly": "wind_speed_10m,wind_direction_10m",
        "models": MODEL,
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "past_days": 1,
        "forecast_days": 3,
    }


def api_get(params):
    """GET dengan retry untuk 429 'minutely'/5xx; pindah endpoint bila 400."""
    global _endpoint_idx
    for attempt in range(6):
        if time.time() > DEADLINE:
            raise RuntimeError(f"melewati batas waktu {MAX_RUNTIME_S // 60} menit, berhenti")
        url = ENDPOINTS[_endpoint_idx]
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            print(f"  koneksi gagal ({e}), ulang...")
            sleep_checked(10 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()

        body = r.text[:300]
        print(f"  HTTP {r.status_code} dari {url}: {body}")

        if r.status_code == 400 and _endpoint_idx + 1 < len(ENDPOINTS):
            _endpoint_idx += 1
            print(f"  pindah ke endpoint {ENDPOINTS[_endpoint_idx]}")
            continue
        if r.status_code == 429:
            low = body.lower()
            if "hourly" in low or "daily" in low:
                raise QuotaExceeded(body)
            wait = int(r.headers.get("Retry-After", 65)) if r.headers.get("Retry-After", "").isdigit() else 65
            print(f"  kena batas per menit, tunggu {wait}s...")
            sleep_checked(wait)
            continue
        if r.status_code >= 500:
            sleep_checked(15 * (attempt + 1))
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {body}")
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
        data = api_get(build_params(chunk))
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
            sleep_checked(PAUSE_S)
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

    # Preflight: 1 titik, gagal cepat kalau kuota habis / endpoint salah
    print("Preflight (1 titik)...")
    try:
        api_get(build_params([(-5.0, 119.5)]))
        print("  preflight OK")
    except QuotaExceeded as e:
        print(f"::error::Kuota Open-Meteo habis, tidak ada gunanya lanjut: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"::error::Preflight gagal: {e}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_ok = 0
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
            n_ok += 1
        except Exception as e:      # file lama dibiarkan utuh bila gagal
            print(f"::warning::GAGAL grid '{tier}': {e}")

    if n_ok == 0:
        print("::error::Semua grid gagal")
        sys.exit(1)
    print(f"Selesai: {n_ok}/{len(TIERS)} grid berhasil")


if __name__ == "__main__":
    main()
