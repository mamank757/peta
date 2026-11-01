/* waves-connect.js — menghubungkan keluaran build_waves.py (data/waves/) ke dashboard.
   Pasang SETELAH <script> utama:  <script src="waves-connect.js"></script>  */
(() => {
const BASE = (typeof WIND_DATA_BASE !== 'undefined' ? WIND_DATA_BASE : './data/') + 'waves/';
const WAVES = ['mjo', 'kelvin', 'rossby'];
const VMAX = { mjo: 30, kelvin: 15, rossby: 15 };
const COL = { mjo: '#00d4ff', kelvin: '#00ccff', rossby: '#cc66ff' };
const NAME = { mjo: 'MJO', kelvin: 'Kelvin Wave', rossby: 'Rossby Wave (ER n=1)' };
const OLR_GRAD = 'linear-gradient(to right,#08306b,#4292c6,#fff,#fb6a4a,#a50f15)';
const store = {};
let imgLayer = null, vecLayer = null;
const $ = id => document.getElementById(id);
const n = (x, d = 1) => (typeof x === 'number' && isFinite(x)) ? x.toFixed(d) : '—';
const sg = (x, d = 1) => n(x, d) === '—' ? '—' : (x >= 0 ? '+' : '') + x.toFixed(d);
const esc = s => String(s ?? '—').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const windOk = m => m.vektor && m.tanggalAngin && Math.abs(Date.parse(m.tanggal) - Date.parse(m.tanggalAngin)) <= 3 * 864e5;
const fl = lon => lon > 180 ? (360 - lon) + '°BB' : lon + '°BT';

/* ---------- UI tambahan (disuntik, HTML asli tidak perlu diubah) ---------- */
document.head.insertAdjacentHTML('beforeend', `<style>
.forecast:not(.collapsed) .collapsible-content{max-height:min(70dvh,780px);overflow-y:auto}
.wlsw{display:flex;gap:12px;justify-content:center;margin-top:6px;font-size:.68rem;color:#fff}
.wlsw label{display:flex;align-items:center;gap:4px;cursor:pointer}.wlsw input{accent-color:#00ff88}
.hov-wrap{position:relative;background:#0b162c;border-radius:8px;overflow:hidden;line-height:0}
.hov-wrap img{width:100%;height:150px;display:block}
.hov-box{position:absolute;top:0;bottom:0;border:1px dashed rgba(0,255,136,.85);border-width:0 1px;pointer-events:none}
.hov-d{position:absolute;left:4px;font-size:.58rem;line-height:1;color:rgba(255,255,255,.75)}
@media(max-width:768px){.forecast:not(.collapsed) .collapsible-content{max-height:min(calc(100dvh - 330px),780px)}body.panel-open .wave-info-badge,body.panel-open .wave-source-badge{display:none!important}body.panel-open.wave-mode .leaflet-control-zoom{display:none}}
.hov-ax{display:flex;justify-content:space-between;font-size:.58rem;color:rgba(255,255,255,.5);margin-top:3px}
</style>`);
$('wavePhaseChip').insertAdjacentHTML('afterend',
  `<div class="wlsw"><label><input type="checkbox" id="wtOlr" checked> OLR tersaring</label>
   <label><input type="checkbox" id="wtVec" checked> Angin 850 hPa (anomali)</label></div>`);
$('gasWavePanel').insertAdjacentHTML('beforeend',
  `<div class="gas-wave-row"><span class="gas-wave-label">📉 Minimum 7 Hari</span><span class="gas-wave-value" id="gwPeak">—</span></div>
   <div class="gas-wave-row"><span class="gas-wave-label">🎯 Keandalan Tepi</span><span class="gas-wave-value" id="gwRel">—</span></div>`);
$('waveChart').parentElement.parentElement.insertAdjacentHTML('afterend',
  `<div id="hovBlock" style="margin-top:10px;display:none">
     <div style="font-size:.62rem;color:rgba(255,255,255,.55);margin-bottom:4px">Hovmöller 90 hari — OLR tersaring (biru = konvektif, merah = kering)</div>
     <div class="hov-wrap"><img id="hovImg" alt="Hovmöller"><div class="hov-box" id="hovBox"></div>
       <span class="hov-d" id="hovT0" style="top:3px"></span><span class="hov-d" id="hovT1" style="bottom:3px"></span></div>
     <div class="hov-ax"><span id="hovLon0"></span><span>garis putus: Indonesia 95–141°BT</span><span id="hovLon1"></span></div>
   </div>`);

$('wdImpact').parentElement.style.display = 'none';   // duplikat baris 'Dampak' di panel atas

/* Ganti sumber lama: tile OWM pendukung -> lapisan kosong; judul panel; legenda */
WAVES.forEach(w => { waveOverlays[w] = () => L.layerGroup(); });
Object.assign(panelTitles, {
  mjo: '🌀 MJO · OLR 30–96 hari',
  kelvin: '〰️ Kelvin Wave · OLR',
  rossby: '🌊 Rossby (ER n=1) · OLR'
});
const lg0 = window.updateLegendUI;
window.updateLegendUI = mode => {
  if (!WAVES.includes(mode)) return lg0(mode);
  const v = store[mode]?.meta?.vmax ?? VMAX[mode], g = $('legendGradient'), s = $('legendScale');
  g.style.display = ''; g.style.background = OLR_GRAD; s.style.display = '';
  s.innerHTML = [`−${v} W/m² (konvektif)`, '0', `+${v} W/m² (tertekan)`].map(t => `<span>${t}</span>`).join('');
};
const sw0 = window.switchRadarLayer;
window.switchRadarLayer = (m, el) => { clearLayers(); document.body.classList.toggle('wave-mode', WAVES.includes(m)); sw0(m, el); };

/* ---------- data ---------- */
const valid = (w, m) => m && m.gelombang === w && Array.isArray(m.bounds) && m.bounds.length === 2 && m.tanggal &&
  Number.isFinite(m.zKalibrasi) && Number.isFinite(m.nilaiBox) && ['konvektif', 'tertekan', 'netral'].includes(m.fase);

async function pack(w) {
  const c = store[w];
  if (c && Date.now() - c.t < 30 * 60e3) return c.meta;
  try {
    const r = await fetch(`${BASE}${w}.json?t=${Date.now()}`, { cache: 'no-store' });
    if (!r.ok) throw new Error(`${w}.json HTTP ${r.status}`);
    const meta = await r.json();
    if (!valid(w, meta)) throw new Error(`${w}.json tidak valid`);
    store[w] = { meta, t: Date.now() };
    return meta;
  } catch (e) { if (c) return c.meta; throw e; }       // pakai data lama bila unduhan gagal
}

/* ---------- lapisan peta ---------- */
function clearLayers() { imgLayer?.remove(); vecLayer?.remove(); imgLayer = vecLayer = null; }

async function layers(w, m) {
  const key = w + m.tanggal, olr = $('wtOlr').checked, vec = $('wtVec').checked && windOk(m);
  if (imgLayer && (imgLayer._k !== key || !olr)) { imgLayer.remove(); imgLayer = null; }
  if (olr && !imgLayer) {
    imgLayer = L.imageOverlay(`${BASE}${w}.png?v=${m.tanggal}`, m.bounds, { opacity: .9, zIndex: 5, interactive: false }).addTo(map);
    imgLayer._k = key;
  }
  const vk = w + m.tanggalAngin;
  if (vecLayer && (vecLayer._k !== vk || !vec)) { vecLayer.remove(); vecLayer = null; }
  if (!vec || vecLayer) return;
  try {
    const r = await fetch(`${BASE}${w}-vec.json?v=${m.tanggalAngin}`, { cache: 'no-cache' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    if (activeLayerName !== w || vecLayer || !$('wtVec').checked) return;
    vecLayer = L.velocityLayer({
      displayValues: false, data, maxVelocity: 5, velocityScale: 0.02, particleAge: 90, lineWidth: 1.5,
      colorScale: ['#ffffff', '#d6f0ff', '#fff2a8', '#ffd166', '#ff9f1c', '#ff5d5d']
    }).addTo(map);
    vecLayer._k = vk;
  } catch (e) { console.warn(`[${w}] vektor tidak dimuat:`, e.message); }
}
['wtOlr', 'wtVec'].forEach(id => $(id).addEventListener('change', () => {
  const m = store[activeLayerName]?.meta; if (m) layers(activeLayerName, m);
}));

/* ---------- panel ---------- */
function render(w, m) {
  const col = COL[w], z = m.zKalibrasi, rel = m.keandalan, t7 = m.ringkas7hari || { nilai: [] };
  const fase = { konvektif: 'Konvektif (basah)', tertekan: 'Tertekan (kering)', netral: 'Netral' }[m.fase];
  const dampak = (m.fase === 'konvektif' && !m.aktif)
    ? 'Sinyal konvektif belum terkonfirmasi (keandalan/propagasi belum cukup) — bukan dasar prakiraan hujan'
    : m.dampak;
  const age = (Date.now() - Date.parse(m.tanggal)) / 864e5;

  $('wtVec').disabled = !windOk(m); $('wtVec').parentElement.style.opacity = windOk(m) ? '' : '.45';
  const sb = $('waveSourceBadge'); sb.style.display = 'flex';
  if (age > 6 || rel === 'rendah') {
    sb.className = 'wave-source-badge fallback';
    sb.innerHTML = age > 6 ? `⚠️ NOAA OLR · data ${Math.floor(age)} hari lalu` : '⚠️ NOAA OLR · keandalan tepi rendah';
  } else { sb.className = 'wave-source-badge gas'; sb.innerHTML = `✅ NOAA OLR · filter W-K · ${esc(m.tanggal)}`; }

  $('waveInfoText').innerHTML = `${NAME[w]} — <b>${esc(m.label)}</b>: OLR tersaring <b>${sg(m.nilaiBox)} W/m²</b> (z ${sg(z, 2)}σ) <span style="color:#00ff88;font-size:.7em">[NOAA]</span>.<br><span>${esc(dampak)}</span>`;
  const c = m.aktif ? col : '#888888';
  $('wavePhaseChip').innerHTML = `<span style="display:inline-block;text-align:center;background:${c}22;border:1px solid ${c}55;color:${c};padding:3px 12px;border-radius:6px;font-size:.72rem;font-weight:700">${NAME[w]}: ${m.aktif ? 'AKTIF' : 'TIDAK AKTIF'} · keandalan ${esc(rel)}</span>`;

  $('gasWavePanel').className = `gas-wave-info active ${w}-style`;
  const set = (id, t, cls) => { const e = $(id); e.className = 'gas-wave-value ' + (cls || w + '-color'); e.textContent = t; };
  set('gwSource', 'NOAA OLR CDR + NCEP R1 · W-K');
  set('gwTanggal', m.tanggal + (m.tanggalAngin && m.tanggalAngin !== m.tanggal ? ` (angin ${m.tanggalAngin}${windOk(m) ? '' : ' · usang, tidak dipakai'})` : ''));
  $('gwMetricLabel').textContent = '🌀 Status & Label'; set('gwMetric', m.label);
  $('gwDetailLabel').textContent = '📊 OLR Tersaring (Indonesia)';
  set('gwDetail', `${sg(m.nilaiBox)} W/m² · z ${sg(z, 2)}σ (±${n(m.galatZ, 2)}σ)`);
  set('gwPeak', `${sg(t7.minimum)} W/m² (z ${sg(t7.zMinimum, 2)}σ)` + (t7.hariLalu == null ? '' : t7.hariLalu === 0 ? ' · hari ini' : ` · ${t7.hariLalu} hari lalu`));
  set('gwRel', `${rel} (r=${n(m.keandalanTepi?.korelasi, 2)})`, rel === 'baik' ? 'safe' : 'warn');
  set('gwArah', m.arah); set('gwImpact', dampak, m.aktif ? 'warn' : 'safe');

  const pc = m.fase === 'konvektif' ? col : m.fase === 'tertekan' ? '#ff9944' : '#888888';
  $('wdPhase').innerText = `${m.aktif ? 'Aktif' : 'Tidak aktif'} — ${fase}`;
  $('wdAmplitude').innerText = `${sg(z, 2)}σ (${sg(m.nilaiBox)} W/m²)`;
  $('wdOLR').innerText = windOk(m)
    ? `u850 ${sg(m.anomaliU850, 2)} · v850 ${sg(m.anomaliAngin, 2)} m/s · ζ ${sg(m.vortisitas, 2)}×10⁻⁵ s⁻¹`
    : `korelasi propagasi r=${n(m.korelasiPropagasi, 2)}`;
  $('wdImpact').innerText = dampak;
  $('wdPhaseFill').style.width = Math.min(Math.abs(z) / 2, 1) * 100 + '%';
  $('wdPhaseFill').style.background = pc;

  const vals = t7.nilai || [], d0 = Date.parse(m.tanggal);
  const labels = vals.map((_, i) => new Date(d0 - (vals.length - 1 - i) * 864e5).toISOString().slice(5, 10).replace('-', '/'));
  if (waveChartInstance) waveChartInstance.destroy();
  const tick = { color: 'rgba(255,255,255,.5)', font: { size: 9 } }, grid = { color: 'rgba(255,255,255,.04)' };
  waveChartInstance = new Chart($('waveChart').getContext('2d'), {
    type: 'bar',
    data: { labels, datasets: [{ data: vals, borderColor: col, borderWidth: 1, borderRadius: 3,
      backgroundColor: vals.map(v => v <= 0 ? col + 'cc' : 'rgba(255,153,68,.65)') }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false },
      title: { display: true, text: 'OLR tersaring 7 hari (W/m²) — negatif = konvektif', color: 'rgba(255,255,255,.55)', font: { size: 9, weight: '400' }, padding: { bottom: 4 } } },
      scales: { x: { ticks: tick, grid }, y: { ticks: tick, grid } } }
  });

  const h = m.hov;
  $('hovBlock').style.display = h ? '' : 'none';
  if (h) {
    const sp = h.lon1 - h.lon0;
    $('hovImg').src = `${BASE}${w}-hov.png?v=${m.tanggal}`;
    $('hovBox').style.left = (95 - h.lon0) / sp * 100 + '%';
    $('hovBox').style.width = 46 / sp * 100 + '%';
    $('hovT0').textContent = h.tanggal0; $('hovT1').textContent = h.tanggal1;
    $('hovLon0').textContent = fl(h.lon0); $('hovLon1').textContent = fl(h.lon1);
  }
  $('waveLoadingIndicator').classList.remove('active');
  $('waveDataPanel').classList.add('active');
}

/* Menggantikan loadWaveData versi GAS/proxy */
window.loadWaveData = async function (w, silent) {
  const ld = $('waveLoadingIndicator'), B = $('waveInfoBadge');
  if (!silent) {
    ld.classList.add('active');
    $('waveDataPanel').classList.remove('active'); $('gasWavePanel').classList.remove('active');
    Object.assign(B.style, { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' });
    $('waveInfoText').innerHTML = '<span class="pulse">⏳ Memuat data gelombang...</span>';
    $('wavePhaseChip').innerHTML = '';
  }
  try {
    const m = await pack(w);
    if (activeLayerName !== w) return;
    render(w, m); layers(w, m); window.updateLegendUI(w);
  } catch (e) {
    if (activeLayerName !== w) return;
    console.warn(`[${w}] gagal:`, e.message);
    ld.classList.remove('active');
    $('waveDataPanel').classList.remove('active'); $('gasWavePanel').classList.remove('active');
    const sb = $('waveSourceBadge'); sb.className = 'wave-source-badge fallback'; sb.style.display = 'flex';
    sb.innerHTML = '⚠️ Data tidak tersedia';
    $('waveInfoText').innerHTML = `<span style="color:#ffcc00">⚠️ Data ${NAME[w]} tidak tersedia (${esc(e.message)}). Status tidak diestimasi dari cuaca lokal.</span>`;
    $('wavePhaseChip').innerHTML = '';
  }
};
setInterval(() => { if (WAVES.includes(activeLayerName)) window.loadWaveData(activeLayerName, true); }, 30 * 60e3);
})();
