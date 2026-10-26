import requests
import xarray as xr
import json
import datetime
import os

def get_latest_gfs_url():
    # Coba cari data hari ini, lalu mundur ke kemarin jika belum ada
    for day_offset in [0, 1, 2]:
        date_target = datetime.datetime.utcnow() - datetime.timedelta(days=day_offset)
        date_str = date_target.strftime("%Y%m%d")

        # Coba siklus waktu dari yang terbaru ke yang terlama
        for cycle in ["18", "12", "06", "00"]:
            url = (
                f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
                f"file=gfs.t{cycle}z.pgrb2.0p25.f000&"
                f"lev_10_m_above_ground=on&var_UGRD=on&var_VGRD=on&"
                f"subregion=&leftlon=115&rightlon=125&toplat=5&bottomlat=-10&"
                f"dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos"
            )
            print(f"Mengecek ketersediaan data: {date_str} cycle {cycle}z...")
            
            # Lakukan request HEAD untuk mengecek eksistensi file tanpa mengunduhnya dulu
            try:
                resp = requests.head(url, timeout=10)
                if resp.status_code == 200:
                    print(f"Data ditemukan di siklus {cycle}z!")
                    return url
            except requests.RequestException:
                continue
                
    return None

url = get_latest_gfs_url()

if not url:
    print("Error: Tidak ada data GFS yang tersedia di server NOAA saat ini.")
    exit(1)

# Download file GRIB2
grib_file = "wind_temp.grib2"
print(f"Mulai mengunduh data GRIB2...")
response = requests.get(url)

if response.status_code == 200:
    with open(grib_file, 'wb') as f:
        f.write(response.content)
    print("Download GRIB2 selesai.")
else:
    print(f"Gagal mengunduh data GFS: {response.status_code}")
    exit(1)

# Buka file GRIB2 menggunakan xarray
ds = xr.open_dataset(grib_file, engine='cfgrib')

# Ekstrak data matriks U dan V (1D array)
u_data = ds['u10'].values.flatten().tolist()
v_data = ds['v10'].values.flatten().tolist()

# Ambil metadata grid
latitudes = ds['latitude'].values.tolist()
longitudes = ds['longitude'].values.tolist()

ny = len(latitudes)
nx = len(longitudes)
la1, la2 = max(latitudes), min(latitudes)
lo1, lo2 = min(longitudes), max(longitudes)
dx = (lo2 - lo1) / (nx - 1)
dy = (la1 - la2) / (ny - 1)

# Susun format JSON leaflet-velocity
leaflet_json = [
    {
        "header": {
            "parameterCategory": 2, "parameterNumber": 2, 
            "nx": nx, "ny": ny,
            "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2,
            "dx": dx, "dy": dy
        },
        "data": u_data
    },
    {
        "header": {
            "parameterCategory": 2, "parameterNumber": 3, 
            "nx": nx, "ny": ny,
            "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2,
            "dx": dx, "dy": dy
        },
        "data": v_data
    }
]

# Simpan hasil parsing
os.makedirs('../data', exist_ok=True)
with open('../data/wind-sulsel.json', 'w') as f:
    json.dump(leaflet_json, f, separators=(',', ':'))

print("Data matriks angin berhasil diproses dan disimpan ke ../data/wind-sulsel.json")
