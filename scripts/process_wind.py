import requests
import xarray as xr
import json
import datetime
import os

# 1. Tentukan waktu run GFS terbaru (00z, 06z, 12z, 18z)
now = datetime.datetime.utcnow()
date_str = now.strftime("%Y%m%d")
cycle = "00" # Untuk produksi, ini bisa diubah dinamis untuk mencari cycle terbaru

# 2. URL NOMADS GFS Filter (Hanya ambil Angin U & V, ketinggian 10m, area Sulawesi/Indonesia)
# Bounding box: Lat 5 to -10, Lon 115 to 125 (Sulawesi & sekitarnya agar ukuran file sangat kecil)
url = (
    f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
    f"file=gfs.t{cycle}z.pgrb2.0p25.f000&"
    f"lev_10_m_above_ground=on&var_UGRD=on&var_VGRD=on&"
    f"subregion=&leftlon=115&rightlon=125&toplat=5&bottomlat=-10&"
    f"dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos"
)

# 3. Download file GRIB2
grib_file = "wind_temp.grib2"
print(f"Mengunduh data dari: {url}")
response = requests.get(url)

if response.status_code == 200:
    with open(grib_file, 'wb') as f:
        f.write(response.content)
    print("Download GRIB2 selesai.")
else:
    print(f"Gagal mengunduh data GFS: {response.status_code}")
    exit(1)

# 4. Buka file GRIB2 menggunakan xarray dan engine cfgrib
ds = xr.open_dataset(grib_file, engine='cfgrib')

# Ekstrak data matriks U dan V, lalu ubah ke array 1D (flatten) agar sesuai dengan struktur leaflet-velocity
u_data = ds['u10'].values.flatten().tolist()
v_data = ds['v10'].values.flatten().tolist()

# Ambil metadata grid (batas koordinat dan resolusi jarak antar titik)
latitudes = ds['latitude'].values.tolist()
longitudes = ds['longitude'].values.tolist()

ny = len(latitudes)
nx = len(longitudes)
la1, la2 = max(latitudes), min(latitudes)
lo1, lo2 = min(longitudes), max(longitudes)
dx = (lo2 - lo1) / (nx - 1)
dy = (la1 - la2) / (ny - 1)

# 5. Susun array tersebut ke dalam format JSON khusus leaflet-velocity
leaflet_json = [
    {
        "header": {
            "parameterCategory": 2, "parameterNumber": 2, # Parameter 2 untuk U-component
            "nx": nx, "ny": ny,
            "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2,
            "dx": dx, "dy": dy
        },
        "data": u_data
    },
    {
        "header": {
            "parameterCategory": 2, "parameterNumber": 3, # Parameter 3 untuk V-component
            "nx": nx, "ny": ny,
            "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2,
            "dx": dx, "dy": dy
        },
        "data": v_data
    }
]

# 6. Simpan hasil parsing ke dalam folder data/
os.makedirs('../data', exist_ok=True)
with open('../data/wind-sulsel.json', 'w') as f:
    json.dump(leaflet_json, f, separators=(',', ':'))

print("Data matriks angin berhasil diproses dan disimpan ke ../data/wind-sulsel.json")
