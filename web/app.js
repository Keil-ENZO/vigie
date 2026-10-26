// Vigie : la carte, le scan, les navires et le rapport. JS sans build.

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const CATEGORY = {
  cargo: { label: "cargo", color: "#38bdf8" },
  tanker: { label: "pétrolier", color: "#f59e0b" },
  passenger: { label: "passagers", color: "#a78bfa" },
  military: { label: "militaire", color: "#ef4444" },
  fishing: { label: "pêche", color: "#34d399" },
  pleasure: { label: "plaisance", color: "#f472b6" },
  service: { label: "service", color: "#facc15" },
  unknown: { label: "inconnu", color: "#94a3b8" },
};
const MAX_SIDE_DEG = 0.5;

const BASEMAPS = {
  sombre: { tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
            attribution: "Esri, HERE, Garmin, © OpenStreetMap" },
  satellite: { tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
               attribution: "Esri, Maxar, Earthstar Geographics" },
};

const state = {
  sensor: "s2",
  aoi: null, aoiName: null, presetId: null, scene: null, scenes: [],
  scan: null, features: [], selected: null, basemap: "sombre", overlay: true, drawing: false,
};

// ---------- utilitaires ----------

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  const type = r.headers.get("content-type") || "";
  return type.includes("json") ? r.json() : r.text();
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmtDate = (iso) => new Date(iso).toLocaleString("fr-FR", { dateStyle: "medium", timeStyle: "short", timeZone: "UTC" }) + " UTC";
const fmtNum = (v, d = 0) => (v == null ? "?" : Number(v).toLocaleString("fr-FR", { maximumFractionDigits: d }));

function markdown(md) {
  // Rendu minimal et sûr : on échappe d'abord, puis titres, gras, listes.
  const lines = esc(md).split("\n");
  let html = "", inList = false;
  for (const raw of lines) {
    const line = raw.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|\s)\*(\S.+?)\*/g, "$1<i>$2</i>");
    const h = line.match(/^(#{1,3})\s+(.*)/);
    const li = line.match(/^\s*[-*]\s+(.*)/);
    if (li) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${li[1]}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    if (h) html += `<h${h[1].length + 1}>${h[2]}</h${h[1].length + 1}>`;
    else if (line.trim()) html += `<p>${line}</p>`;
  }
  return html + (inList ? "</ul>" : "");
}

function bboxPolygon([w, s, e, n]) {
  return { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } };
}

function offset(p, meters, deg) {
  const t = (deg * Math.PI) / 180;
  return [p.lon + (meters * Math.sin(t)) / (111320 * Math.cos((p.lat * Math.PI) / 180)), p.lat + (meters * Math.cos(t)) / 111320];
}

// Navire en route : une flèche dans le sens de marche, levé par le sillage.
// Sinon : son axe seul, ambigu à 180° près.
function headingFeatures(p) {
  if (p.course_deg != null) {
    const len = Math.max(250, p.length_m * 2), tip = offset(p, len, p.course_deg);
    const wing = (d) => offset({ lon: tip[0], lat: tip[1] }, len * 0.28, p.course_deg + 180 + d);
    return [{ type: "Feature", properties: { id: p.id, kind: "course" }, geometry: { type: "MultiLineString",
      coordinates: [[[p.lon, p.lat], tip], [wing(-28), tip, wing(28)]] } }];
  }
  if (p.heading_deg == null || !p.length_m) return [];
  const half = Math.max(p.length_m / 2, 40);
  return [{ type: "Feature", properties: { id: p.id, kind: "axis" }, geometry: { type: "LineString",
    coordinates: [offset(p, half, p.heading_deg + 180), offset(p, half, p.heading_deg)] } }];
}

const category = (p) => (p.analysis && !p.analysis.error ? p.analysis.category : null);
const rejected = (p) => p.analysis && p.analysis.is_vessel === false;
const aisCat = (p) => (p.ais?.status === "matched" ? p.ais.category : null);

// ---------- carte ----------

const map = (window.__vigieMap = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    glyphs: "https://fonts.openmaptiles.org/{fontstack}/{range}.pbf",
    sources: { base: { type: "raster", tiles: BASEMAPS.sombre.tiles, tileSize: 256, attribution: BASEMAPS.sombre.attribution } },
    layers: [{ id: "base", type: "raster", source: "base" }],
  },
  center: [5.93, 43.1], zoom: 11, attributionControl: { compact: true },
}));
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");

map.on("load", () => {
  map.addSource("aoi", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addLayer({ id: "aoi-fill", type: "fill", source: "aoi", paint: { "fill-color": "#38bdf8", "fill-opacity": 0.06 } });
  map.addLayer({ id: "aoi-line", type: "line", source: "aoi", paint: { "line-color": "#38bdf8", "line-width": 1.5, "line-dasharray": [3, 2] } });

  map.addSource("boxes", { type: "geojson", data: { type: "FeatureCollection", features: [] }, promoteId: "id" });
  map.addSource("points", { type: "geojson", data: { type: "FeatureCollection", features: [] }, promoteId: "id" });
  map.addSource("headings", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addSource("ais", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  const color = ["match", ["get", "cat"], ...Object.entries(CATEGORY).flatMap(([k, v]) => [k, v.color]), "#f97316"];
  map.addLayer({ id: "box-line", type: "line", source: "boxes", minzoom: 12,
    paint: { "line-color": color, "line-width": ["case", ["boolean", ["feature-state", "sel"], false], 2.5, 1.2] } });
  map.addLayer({ id: "heading", type: "line", source: "headings", minzoom: 11,
    paint: { "line-color": ["case", ["==", ["get", "kind"], "course"], "#34d399", "#ffffff"],
             "line-width": ["case", ["==", ["get", "kind"], "course"], 1.8, 1], "line-opacity": 0.85 } });
  map.addLayer({ id: "halo", type: "circle", source: "points",
    paint: { "circle-radius": ["case", ["boolean", ["feature-state", "sel"], false], 16, 9], "circle-color": color, "circle-opacity": 0.18,
             "circle-stroke-width": ["case", ["boolean", ["feature-state", "sel"], false], 2, 0], "circle-stroke-color": "#fff" } });
  map.addLayer({ id: "point", type: "circle", source: "points",
    paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 2.5, 14, 4], "circle-color": color,
             "circle-opacity": ["case", ["get", "rejected"], 0.35, 1] } });
  map.addLayer({ id: "ais-missed", type: "circle", source: "ais",
    paint: { "circle-radius": 4, "circle-color": "rgba(0,0,0,0)", "circle-stroke-width": 1.3, "circle-stroke-color": "#cbd5e1", "circle-stroke-opacity": 0.75 } });
  map.addLayer({ id: "ais-label", type: "symbol", source: "ais", minzoom: 13.5,
    layout: { "text-field": ["get", "label"], "text-size": 10, "text-offset": [0, 1.2], "text-font": ["Open Sans Regular"] },
    paint: { "text-color": "#94a3b8", "text-halo-color": "#0b1220", "text-halo-width": 1.2 } });
  map.addLayer({ id: "dark-ring", type: "circle", source: "points", filter: ["==", ["get", "dark"], true],
    paint: { "circle-radius": 13, "circle-color": "rgba(0,0,0,0)", "circle-stroke-width": 2, "circle-stroke-color": "#f97316" } });
  map.addLayer({ id: "label", type: "symbol", source: "points", minzoom: 13,
    layout: { "text-field": ["get", "label"], "text-size": 11, "text-offset": [0, 1.3], "text-font": ["Open Sans Regular"] },
    paint: { "text-color": "#e2e8f0", "text-halo-color": "#0b1220", "text-halo-width": 1.5 } });

  // AIS en direct : flèche si le navire avance, point s'il est arrêté. Icônes SDF, colorées par type.
  const icon = (draw) => {
    const n = 32, c = document.createElement("canvas");
    c.width = c.height = n;
    const g = c.getContext("2d");
    g.fillStyle = "#fff";
    draw(g, n);
    return { width: n, height: n, data: g.getImageData(0, 0, n, n).data };
  };
  map.addImage("live-arrow", icon((g, n) => { g.beginPath(); g.moveTo(n / 2, 3); g.lineTo(n - 7, n - 4); g.lineTo(n / 2, n - 10); g.lineTo(7, n - 4); g.closePath(); g.fill(); }), { sdf: true });
  map.addImage("live-dot", icon((g, n) => { g.beginPath(); g.arc(n / 2, n / 2, n / 4, 0, 2 * Math.PI); g.fill(); }), { sdf: true });
  map.addSource("ais-live", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addLayer({ id: "ais-live-ring", type: "circle", source: "ais-live",
    paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 7, 13, 13], "circle-color": "rgba(56,189,248,0.12)",
             "circle-stroke-width": 1.5, "circle-stroke-color": "#38bdf8" } });
  map.addLayer({ id: "ais-live", type: "symbol", source: "ais-live",
    layout: { "icon-image": ["case", ["get", "moving"], "live-arrow", "live-dot"], "icon-size": ["interpolate", ["linear"], ["zoom"], 6, 0.55, 13, 0.95],
              "icon-rotate": ["get", "rot"], "icon-rotation-alignment": "map", "icon-allow-overlap": true },
    // Type encore inconnu : blanc, pour trancher sur le violet du radar comme sur la mer sombre.
    paint: { "icon-color": ["match", ["get", "cat"], ...Object.entries(CATEGORY).filter(([k]) => k !== "unknown").flatMap(([k, v]) => [k, v.color]), "#ffffff"],
             "icon-halo-color": "#0b1220", "icon-halo-width": 2.5 } });
  map.addLayer({ id: "ais-live-label", type: "symbol", source: "ais-live", minzoom: 12,
    layout: { "text-field": ["get", "name"], "text-size": 10.5, "text-offset": [0, 1.3], "text-font": ["Open Sans Regular"] },
    paint: { "text-color": "#e2e8f0", "text-halo-color": "#0b1220", "text-halo-width": 1.4 } });
  // Tout le cercle est cliquable, pas seulement l'icône du centre.
  for (const layer of ["ais-live-ring", "ais-live"]) {
    map.on("mouseenter", layer, () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", layer, () => (map.getCanvas().style.cursor = ""));
  }
  map.on("click", (e) => {
    const pad = 6, box = [[e.point.x - pad, e.point.y - pad], [e.point.x + pad, e.point.y + pad]];
    const hit = map.queryRenderedFeatures(box, { layers: ["ais-live-ring", "ais-live"] })[0];
    if (hit) liveAisPopup(hit);
  });

  map.on("click", "halo", (e) => select(e.features[0].properties.id, false));
  map.on("mouseenter", "halo", () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", "halo", () => (map.getCanvas().style.cursor = ""));
  init();
});

function setAoi(bbox, name = null, presetId = null) {
  state.aoi = bbox.map((v) => Math.round(v * 1e5) / 1e5);
  state.aoiName = name; state.presetId = presetId; state.scene = null;
  map.getSource("aoi").setData(bboxPolygon(state.aoi));
  $$(".preset").forEach((b) => b.classList.toggle("active", b.dataset.id === presetId));
  const [w, s, e, n] = state.aoi;
  const km = (d, lat) => (d * 111.32 * Math.cos((lat * Math.PI) / 180)).toFixed(1);
  $("#aoi-label").textContent = `${name ? name + " · " : ""}${km(e - w, (s + n) / 2)} × ${((n - s) * 111.32).toFixed(1)} km`;
  loadScenes();
}

// Dessin d'une zone au glisser.
let drawStart = null;
$("#draw-btn").onclick = () => {
  state.drawing = !state.drawing;
  $("#draw-btn").classList.toggle("on", state.drawing);
  $("#draw-btn").textContent = state.drawing ? "Glisse sur la carte…" : "Dessiner une zone";
  map.getCanvas().style.cursor = state.drawing ? "crosshair" : "";
  state.drawing ? map.dragPan.disable() : map.dragPan.enable();
};
map.on("mousedown", (e) => { if (state.drawing) drawStart = e.lngLat; });
map.on("mousemove", (e) => {
  if (!drawStart) return;
  const b = [Math.min(drawStart.lng, e.lngLat.lng), Math.min(drawStart.lat, e.lngLat.lat), Math.max(drawStart.lng, e.lngLat.lng), Math.max(drawStart.lat, e.lngLat.lat)];
  map.getSource("aoi").setData(bboxPolygon(b));
});
map.on("mouseup", (e) => {
  if (!drawStart) return;
  const b = [Math.min(drawStart.lng, e.lngLat.lng), Math.min(drawStart.lat, e.lngLat.lat), Math.max(drawStart.lng, e.lngLat.lng), Math.max(drawStart.lat, e.lngLat.lat)];
  drawStart = null;
  $("#draw-btn").click();
  if (b[2] - b[0] < 0.005 || b[3] - b[1] < 0.005) return;
  setAoi(clampBox(b));
});

function clampBox([w, s, e, n]) {
  const cx = (w + e) / 2, cy = (s + n) / 2, hw = Math.min((e - w) / 2, MAX_SIDE_DEG / 2), hh = Math.min((n - s) / 2, MAX_SIDE_DEG / 2);
  return [cx - hw, cy - hh, cx + hw, cy + hh];
}

$("#view-btn").onclick = () => {
  const b = map.getBounds();
  setAoi(clampBox([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]));
};

$("#basemap-btn").onclick = () => {
  state.basemap = state.basemap === "sombre" ? "satellite" : "sombre";
  const src = map.getSource("base");
  src.setTiles(BASEMAPS[state.basemap].tiles);
  $("#basemap-btn").textContent = `Fond : ${state.basemap}`;
};
$("#overlay-btn").onclick = () => {
  state.overlay = !state.overlay;
  if (map.getLayer("scan-img")) map.setLayoutProperty("scan-img", "visibility", state.overlay ? "visible" : "none");
  $("#overlay-btn").textContent = `Image Sentinel : ${state.overlay ? "visible" : "masquée"}`;
};

// ---------- AIS en direct ----------

let liveTimer = null;
const NAV = ["en route", "au mouillage", "non manœuvrant", "manœuvre restreinte", "contraint par son tirant d'eau",
  "amarré", "échoué", "en pêche", "à la voile"];

async function liveAisTick() {
  const b = map.getBounds();
  const w = Math.max(-180, b.getWest()), e = Math.min(180, b.getEast()), s = Math.max(-85, b.getSouth()), n = Math.min(85, b.getNorth());
  const btn = $("#live-ais-btn");
  if (e - w > 5 || n - s > 5) { btn.textContent = "AIS en direct : zoome"; return; }
  try {
    const r = await api(`/api/ais-live?west=${w}&south=${s}&east=${e}&north=${n}`);
    const st = r.state;
    if (!st.configured) { btn.textContent = "AIS en direct : clé absente"; return; }
    map.getSource("ais-live").setData({ type: "FeatureCollection", features: r.vessels.map((v) => ({
      type: "Feature", geometry: { type: "Point", coordinates: [v.lon, v.lat] },
      properties: { ...v, cat: v.category || "unknown", name: v.name || v.mmsi, moving: (v.sog_kn ?? 0) >= 0.5,
                    rot: v.cog_deg ?? v.heading_deg ?? 0 } })) });
    btn.textContent = st.status === "connecté" ? `AIS en direct : ${r.vessels.length} navires` : `AIS en direct : ${st.status}`;
    btn.title = st.error || `${st.messages} messages reçus`;
  } catch (err) {
    btn.textContent = "AIS en direct : erreur";
    btn.title = err.message;
  }
}

function liveAisPopup(f) {
  const p = f.properties, cat = CATEGORY[p.cat];
  const ago = p.age_s < 90 ? `il y a ${p.age_s} s` : `il y a ${Math.round(p.age_s / 60)} min`;
  popup?.remove();
  popup = new maplibregl.Popup({ offset: 10, maxWidth: "300px" }).setLngLat(f.geometry.coordinates).setHTML(`
    <div><b style="color:var(--accent)">${esc(p.name)}</b>${cat && p.cat !== "unknown" ? ` · ${cat.label}` : ""}
    <p style="margin:4px 0 0;font-size:12px">MMSI ${esc(p.mmsi)}${p.length_m && p.length_m !== "null" ? ` · ${p.length_m} m` : ""}</p>
    <p style="margin:2px 0 0;font-size:12px">${p.sog_kn ?? "?"} nd · route ${p.cog_deg != null ? Math.round(p.cog_deg) + "°" : "?"}${NAV[p.nav_status] ? ` · ${NAV[p.nav_status]}` : ""}</p>
    ${p.destination && p.destination !== "null" ? `<p style="margin:2px 0 0;font-size:12px">vers ${esc(p.destination)}</p>` : ""}
    <p style="margin:2px 0 0;font-size:11px;color:var(--muted)">AIS en direct, ${ago}</p></div>`).addTo(map);
}

$("#live-ais-btn").onclick = () => {
  if (liveTimer) {
    clearInterval(liveTimer); liveTimer = null;
    map.getSource("ais-live").setData({ type: "FeatureCollection", features: [] });
    $("#live-ais-btn").textContent = "AIS en direct : off";
    $("#live-ais-btn").classList.remove("on");
    return;
  }
  $("#live-ais-btn").classList.add("on");
  $("#live-ais-btn").textContent = "AIS en direct : connexion…";
  liveAisTick();
  liveTimer = setInterval(liveAisTick, 5000);
};

// ---------- passes et scan ----------

async function loadScenes() {
  const box = $("#scenes");
  $("#scan-btn").disabled = true;
  box.innerHTML = `<p class="hint">Recherche des passes…</p>`;
  const [w, s, e, n] = state.aoi;
  try {
    state.scenes = state.sensor === "s1"
      ? await api(`/api/scenes-sar?west=${w}&south=${s}&east=${e}&north=${n}&days=${$("#days").value}`)
      : await api(`/api/scenes?west=${w}&south=${s}&east=${e}&north=${n}&days=${$("#days").value}&max_cloud=${$("#cloud").value}`);
  } catch (err) {
    box.innerHTML = `<p class="hint">Erreur : ${esc(err.message)}</p>`;
    return;
  }
  if (!state.scenes.length) { box.innerHTML = `<p class="hint">Aucune passe. Élargis la fenêtre ou le seuil de nuages.</p>`; return; }
  box.innerHTML = state.scenes.map((sc, i) =>
    `<div class="scene${i === 0 ? " active" : ""}" data-i="${i}"><span>${fmtDate(sc.datetime)} <small class="hint">${esc(sc.platform.replace("sentinel-", "S"))}</small></span><span class="cloud">${state.sensor === "s1" ? `${esc(sc.orbit_state === "ascending" ? "montante" : "descendante")}` : `☁ ${sc.cloud_cover.toFixed(1)} %`}</span></div>`).join("");
  state.scene = state.scenes[0];
  $$(".scene").forEach((el) => (el.onclick = () => {
    $$(".scene").forEach((x) => x.classList.remove("active"));
    el.classList.add("active");
    state.scene = state.scenes[+el.dataset.i];
  }));
  $("#scan-btn").disabled = false;
}
$$("#sensor button").forEach((b) => (b.onclick = () => {
  state.sensor = b.dataset.sensor;
  $$("#sensor button").forEach((x) => x.classList.toggle("active", x === b));
  const radar = state.sensor === "s1";
  $("#pass-title").textContent = radar ? "Passe Sentinel-1" : "Passe Sentinel-2";
  $("#sensor-hint").textContent = radar
    ? "Radar en bande C : voit la nuit et à travers les nuages. Détection CFAR, sans IA, sur CPU."
    : "Image en vraies couleurs, de jour et sans nuages.";
  $("#cloud-label").classList.toggle("hidden", radar);
  $("#conf").closest("label").classList.toggle("hidden", radar);
  $("#do-analyse").closest("label").classList.toggle("hidden", radar);
  if (state.aoi) loadScenes();
}));
$("#days").onchange = $("#cloud").onchange = () => state.aoi && loadScenes();
$("#conf").oninput = () => ($("#conf-out").textContent = (+$("#conf").value).toFixed(2));

$("#scan-btn").onclick = async () => {
  $("#scan-btn").disabled = true;
  try {
    const job = await api(state.sensor === "s1" ? "/api/scans-sar" : "/api/scans", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bbox: state.aoi, name: state.aoiName, scene_id: state.scene?.id, confidence: +$("#conf").value, analyse: state.sensor === "s2" && $("#do-analyse").checked }),
    });
    const done = await follow(job.id);
    if (done) await openScan(done.result.scan_id);
  } finally {
    $("#scan-btn").disabled = false;
  }
};

async function follow(jobId) {
  const box = $("#job");
  box.classList.remove("hidden", "error");
  for (;;) {
    const j = await api(`/api/jobs/${jobId}`);
    $("#job-bar").style.width = `${Math.round(j.progress * 100)}%`;
    $("#job-msg").textContent = j.message;
    if (j.status === "done") { setTimeout(() => box.classList.add("hidden"), 1500); return j; }
    if (j.status === "error") { box.classList.add("error"); return null; }
    await new Promise((r) => setTimeout(r, 700));
  }
}

// ---------- affichage d'un scan ----------

async function openScan(id) {
  const data = await api(`/api/scans/${id}`);
  state.scan = data;
  state.features = data.detections.features.filter((f) => !f.properties.fixed);
  const { meta } = data;

  if (map.getLayer("scan-img")) map.removeLayer("scan-img");
  if (map.getSource("scan-img")) map.removeSource("scan-img");
  map.addSource("scan-img", { type: "image", url: `/api/scans/${id}/image.jpg`, coordinates: meta.corners });
  map.addLayer({ id: "scan-img", type: "raster", source: "scan-img", paint: { "raster-opacity": 0.95, "raster-fade-duration": 0 },
                 layout: { visibility: state.overlay ? "visible" : "none" } }, "aoi-fill");

  const withCat = (f) => ({ ...f, properties: { ...f.properties, cat: category(f.properties) ?? aisCat(f.properties) ?? "none",
    rejected: !!rejected(f.properties), dark: f.properties.ais?.status === "dark",
    label: f.properties.ais?.name ? `${f.properties.id} · ${f.properties.ais.name}` : f.properties.id } });
  map.getSource("boxes").setData({ type: "FeatureCollection", features: state.features.map(withCat) });
  map.getSource("points").setData({ type: "FeatureCollection", features: state.features.map((f) => withCat({
    type: "Feature", properties: f.properties, geometry: { type: "Point", coordinates: [f.properties.lon, f.properties.lat] } })) });
  map.getSource("headings").setData({ type: "FeatureCollection", features: state.features.flatMap((f) => headingFeatures(f.properties)) });
  map.getSource("ais").setData({ type: "FeatureCollection", features: (data.detections.ais_only || []).map((v) => ({
    type: "Feature", properties: { label: `${v.name || v.mmsi}${v.length_m ? ` · ${v.length_m} m` : ""}` },
    geometry: { type: "Point", coordinates: [v.lon, v.lat] } })) });
  map.getSource("aoi").setData(bboxPolygon(meta.bbox));
  map.fitBounds([[meta.bbox[0], meta.bbox[1]], [meta.bbox[2], meta.bbox[3]]], { padding: 30, duration: 900 });

  renderResult();
}

function renderResult() {
  const { meta, report_md } = state.scan;
  const ps = state.features.map((f) => f.properties);
  const fixedCount = meta.counts.fixed || 0;
  const kept = ps.filter((p) => !rejected(p));
  // Avec l'AIS, sa vitesse fait foi ; sinon le sillage probable.
  const underway = (p) => (p.ais?.status === "matched" && p.ais.sog_kn != null ? p.ais.sog_kn >= 1 : !!p.underway);
  const moving = kept.filter(underway).length;
  const longest = Math.max(0, ...kept.map((p) => p.length_m || 0));
  $("#empty").classList.add("hidden");
  $("#result").classList.remove("hidden");
  $("#r-title").textContent = meta.name || `Zone ${meta.bbox.map((v) => v.toFixed(2)).join(", ")}`;
  const radar = meta.sensor === "sentinel-1";
  $("#r-sub").textContent = `${radar ? "Radar Sentinel-1" : "Optique Sentinel-2"} · ${fmtDate(meta.scene.datetime)} · ${meta.scene.id}${radar ? "" : ` · ☁ ${meta.scene.cloud_cover.toFixed(1)} %`}`;
  const analysed = ps.some((p) => p.analysis);
  const stats = [
    [kept.length, "navires"],
    ...(radar && !state.scan.detections.ais_summary ? [] : [[moving, state.scan.detections.ais_summary ? "en route, AIS" : "sillage probable"]]),
    [`${fmtNum(longest)} m`, "le plus long"],
    [`${meta.size_px[0]}×${meta.size_px[1]}`, "pixels à 10 m"],
    [`${meta.timing_s.detect.toFixed(1)} s`, radar ? "détection CFAR, CPU" : "détection 3090"],
    ...(radar ? [] : [[analysed ? ps.length - kept.length : "–", "rejetés par VLM"]]),
  ];
  const as = state.scan.detections.ais_summary;
  if (fixedCount) stats.push([fixedCount, "objets fixes écartés"]);
  if (as) stats.push(
    [`${as.matched}/${as.detections}`, "identifiés par AIS"],
    [as.dark, "sombres, sans AIS"],
    [as.recall_ais_50m != null ? `${Math.round(as.recall_ais_50m * 100)} %` : "–", `rappel AIS > 50 m (${as.ais_50m})`]);
  $("#stats").innerHTML = stats.map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");

  $("#r-actions").innerHTML = `
    ${analysed || radar ? "" : `<button class="ghost" id="analyse-btn">Analyser avec le VLM</button>`}
    ${state.scan.ais_available && !state.scan.detections.ais_summary ? `<button class="ghost" id="ais-btn">Croiser avec l'AIS</button>` : ""}
    <a class="ghost" style="flex:1;text-align:center;text-decoration:none;padding:7px;border:1px solid var(--line);border-radius:8px;font-size:12.5px" href="/api/scans/${meta.id}/detections.geojson" download="${meta.id}.geojson">GeoJSON</a>`;
  const ab = $("#analyse-btn");
  if (ab) ab.onclick = async () => {
    ab.disabled = true; ab.textContent = "Analyse en cours…";
    try { const job = await api(`/api/scans/${meta.id}/analyse`, { method: "POST" }); if (await follow(job.id)) await openScan(meta.id); }
    catch (err) { ab.textContent = err.message; }
  };

  const aisBtn = $("#ais-btn");
  if (aisBtn) aisBtn.onclick = async () => {
    aisBtn.disabled = true; aisBtn.textContent = "Lecture de l'AIS…";
    try { const job = await api(`/api/scans/${meta.id}/ais`, { method: "POST" }); if (await follow(job.id)) await openScan(meta.id); }
    catch (err) { aisBtn.textContent = err.message; }
  };

  const sorted = [...ps].sort((a, b) => (rejected(a) - rejected(b)) || (b.length_m || 0) - (a.length_m || 0));
  $("#vessels").innerHTML = sorted.map((p) => {
    const cat = category(p), c = CATEGORY[cat], a = p.ais;
    const aisLine = a?.status === "matched"
      ? `<div class="ais-line"><b>${esc(a.name || a.mmsi)}</b> · ${esc(a.ship_type || "type ?")} · ${a.length_m ? `${fmtNum(a.length_m)} m déclarés` : "longueur ?"} · ${esc(a.nav_status || "")}</div>`
      : a?.status === "dark" ? `<div class="ais-line dark">Aucun AIS à moins de 500 m : navire sombre ou fausse détection</div>` : "";
    return `<div class="vessel${rejected(p) ? " rejected" : ""}${a?.status === "dark" ? " is-dark" : ""}" data-id="${p.id}">
      <img loading="lazy" src="/api/scans/${meta.id}/crops/${p.id}.png" alt="${p.id}">
      <div>
        <div class="top"><span class="id">${p.id}</span>${c ? `<span class="badge" style="background:${c.color}22;color:${c.color}">${c.label}</span>` : ""}</div>
        <div class="meta">${fmtNum(p.length_m)} m${p.course_deg != null ? ` · cap ${fmtNum(p.course_deg)}°` : p.heading_deg != null ? ` · axe ${fmtNum(p.heading_deg)}°` : ""} · conf. ${(p.confidence * 100).toFixed(0)} %
          ${p.underway ? `<span class="moving">● sillage probable</span>` : ""}${rejected(p) ? " · rejeté" : ""}</div>
        ${aisLine}
        ${p.analysis?.note ? `<div class="note">${esc(p.analysis.note)}</div>` : ""}
      </div></div>`;
  }).join("") || `<p class="hint">Aucun navire détecté au-dessus du seuil.</p>`;
  $$(".vessel").forEach((el) => {
    el.onclick = () => select(el.dataset.id, true);
    el.onmouseenter = () => highlight(el.dataset.id);
  });

  $("#report").innerHTML = report_md
    ? `<p class="meta-line">Rédigé par ${esc(state.scan.meta.vlm || "le VLM")} à partir des seules mesures du scan.</p>${markdown(report_md)}`
    : `<p class="hint">Pas encore de rapport. Lance l'analyse.</p>`;
}

$$("#result .seg button").forEach((b) => (b.onclick = () => {
  $$("#result .seg button").forEach((x) => x.classList.toggle("active", x === b));
  for (const v of ["vessels", "report", "activity"]) $(`#${v}`).classList.toggle("hidden", b.dataset.view !== v);
  if (b.dataset.view === "activity") renderActivity();
}));

let highlighted = null, popup = null;
function highlight(id) {
  for (const src of ["points", "boxes"]) {
    if (highlighted) map.setFeatureState({ source: src, id: highlighted }, { sel: false });
    if (id) map.setFeatureState({ source: src, id }, { sel: true });
  }
  highlighted = id;
}

function select(id, fly) {
  const f = state.features.find((x) => x.properties.id === id);
  if (!f) return;
  const p = f.properties;
  highlight(id);
  $$(".vessel").forEach((el) => el.classList.toggle("active", el.dataset.id === id));
  document.querySelector(`.vessel[data-id="${id}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  if (fly) map.flyTo({ center: [p.lon, p.lat], zoom: Math.max(map.getZoom(), 14.5), duration: 800 });
  const cat = category(p);
  popup?.remove();
  popup = new maplibregl.Popup({ offset: 12, maxWidth: "320px" }).setLngLat([p.lon, p.lat]).setHTML(`
    <div class="pop"><img src="/api/scans/${state.scan.meta.id}/crops/${p.id}.png">
    <div><b>${p.id}</b>${cat ? ` · ${CATEGORY[cat].label}` : ""}
      <p>${fmtNum(p.length_m)} m × ${fmtNum(p.width_m)} m${p.course_deg != null ? `, cap ${fmtNum(p.course_deg)}°` : p.heading_deg != null ? `, axe ${fmtNum(p.heading_deg)}°` : ""}</p>
      ${p.wake_m ? `<p>sillage ${fmtNum(p.wake_m)} m : en route</p>` : ""}
      ${p.ais?.status === "matched" ? `<p>AIS : <b>${esc(p.ais.name || p.ais.mmsi)}</b>, ${esc(p.ais.ship_type || "?")}, ${p.ais.length_m ?? "?"} m, ${p.ais.sog_kn ?? "?"} nd, à ${p.ais.distance_m} m</p>` : ""}
      ${p.ais?.status === "dark" ? `<p style="color:#f97316">Sans AIS : navire sombre ?</p>` : ""}
      <p>${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}</p>
      <p>détecteur ${(p.confidence * 100).toFixed(0)} %${p.water_fraction != null ? ` · eau ${(p.water_fraction * 100).toFixed(0)} %` : ""}</p>
      ${p.analysis?.note ? `<p><i>${esc(p.analysis.note)}</i></p>` : ""}</div></div>`).addTo(map);
}

// ---------- activité : la zone sur plusieurs passes ----------

async function renderActivity() {
  const box = $("#activity"), m = state.scan.meta, [w, s, e, n] = m.bbox;
  box.innerHTML = `<p class="hint">Chargement…</p>`;
  const found = await api(`/api/series?west=${w}&south=${s}&east=${e}&north=${n}`);
  if (!found.length) {
    box.innerHTML = `<p class="hint">Vigie peut scanner cette zone sur ses dernières passes claires et compter les arrivées et les départs entre deux passes.</p>
      <button class="ghost" id="series-btn" style="width:100%">Suivre la zone sur 6 passes</button>`;
    $("#series-btn").onclick = async () => {
      $("#series-btn").disabled = true;
      const job = await api("/api/series", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bbox: m.bbox, name: m.name, n: 6, days: 90, max_cloud: 15 }) });
      if (await follow(job.id)) renderActivity();
    };
    return;
  }
  const steps = found[0].steps, max = Math.max(1, ...steps.map((x) => x.vessels));
  const W = 340, H = 170, padL = 26, padB = 26, padT = 18, bw = Math.min(34, (W - padL - 8) / steps.length - 6);
  const x = (i) => padL + 4 + i * ((W - padL - 8) / steps.length) + ((W - padL - 8) / steps.length - bw) / 2;
  const y = (v) => H - padB - (v / max) * (H - padB - padT);
  const ticks = [0, Math.round(max / 2), max];
  const last = steps.length - 1, top = steps.reduce((a, b, i) => (b.vessels > steps[a].vessels ? i : a), 0);
  const bars = steps.map((st, i) => {
    const h = Math.max(2, H - padB - y(st.vessels)), r = Math.min(4, bw / 2, h);
    const x0 = x(i), y0 = y(st.vessels), yb = H - padB;
    // barre ancrée à la base, extrémité de données arrondie à 4 px
    const d = `M${x0},${yb} V${y0 + r} Q${x0},${y0} ${x0 + r},${y0} H${x0 + bw - r} Q${x0 + bw},${y0} ${x0 + bw},${y0 + r} V${yb} Z`;
    const label = i === last || i === top ? `<text x="${x0 + bw / 2}" y="${y0 - 5}" text-anchor="middle" class="v">${st.vessels}</text>` : "";
    const date = new Date(st.datetime).toLocaleDateString("fr-FR", { day: "numeric", month: "short", timeZone: "UTC" });
    return `<g class="bar-g" data-i="${i}"><rect x="${x0 - 3}" y="${padT}" width="${bw + 6}" height="${H - padB - padT}" fill="transparent"/>
      <path d="${d}" class="bar${st.scan_id === m.id ? " cur" : ""}"/>${label}
      <text x="${x0 + bw / 2}" y="${H - 9}" text-anchor="middle" class="t">${date}</text></g>`;
  }).join("");
  box.innerHTML = `
    <h3 class="chart-title">Navires détectés par passe</h3>
    <div class="chart-wrap"><svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="Navires détectés par passe">
      ${ticks.map((t) => `<line x1="${padL}" x2="${W}" y1="${y(t)}" y2="${y(t)}" class="grid"/><text x="${padL - 5}" y="${y(t) + 3}" text-anchor="end" class="t">${t}</text>`).join("")}
      ${bars}</svg><div class="tip hidden" id="chart-tip"></div></div>
    <table class="mini"><thead><tr><th>Passe</th><th>Navires</th><th>Restés</th><th>Arrivés</th><th>Partis</th></tr></thead><tbody>
      ${steps.map((st) => `<tr data-scan="${st.scan_id}"><td>${st.datetime.slice(0, 10)}</td><td>${st.vessels}</td><td>${st.vs_previous?.stayed ?? "–"}</td><td>${st.vs_previous?.arrived ?? "–"}</td><td>${st.vs_previous?.left ?? "–"}</td></tr>`).join("")}
    </tbody></table>
    <p class="hint">Un navire retrouvé à moins de 250 m d'une passe à l'autre est compté comme resté. Clique une barre pour ouvrir la passe.</p>`;
  const tip = $("#chart-tip");
  $$(".bar-g").forEach((g) => {
    const st = steps[+g.dataset.i];
    g.onmouseenter = () => {
      const v = st.vs_previous;
      tip.innerHTML = `<b>${fmtDate(st.datetime)}</b><br>${st.vessels} navires · ☁ ${st.cloud_cover.toFixed(1)} %${v ? `<br>${v.stayed} restés, ${v.arrived} arrivés, ${v.left} partis` : ""}`;
      const bb = g.getBoundingClientRect(), wb = $(".chart-wrap").getBoundingClientRect();
      tip.style.left = `${Math.min(bb.left - wb.left, wb.width - 180)}px`;
      tip.classList.remove("hidden");
    };
    g.onmouseleave = () => tip.classList.add("hidden");
    g.onclick = () => openScan(st.scan_id).then(() => $$("#result .seg button")[2].click());
  });
  $$("tr[data-scan]").forEach((tr) => (tr.onclick = () => openScan(tr.dataset.scan).then(() => $$("#result .seg button")[2].click())));
}

// ---------- historique ----------

async function loadHistory() {
  const list = await api("/api/scans");
  $("#history").innerHTML = list.map((m) => `
    <div class="hist" data-id="${m.id}"><div class="t">${esc(m.name || m.bbox.map((v) => v.toFixed(2)).join(", "))}</div>
    <div class="s">${fmtDate(m.scene.datetime)} · ${m.counts.vessels} navires${m.analysed ? " · analysé" : ""}</div></div>`).join("")
    || `<p class="hint">Aucun scan pour l'instant.</p>`;
  $$(".hist").forEach((el) => (el.onclick = () => openScan(el.dataset.id)));
}

// ---------- image libre ----------

$("#upload-file").onchange = () => ($("#upload-btn").disabled = !$("#upload-file").files.length);
$("#up-conf").oninput = () => ($("#up-conf-out").textContent = (+$("#up-conf").value).toFixed(2));
$("#upload-close").onclick = () => $("#upload-view").classList.add("hidden");
$("#upload-btn").onclick = async () => {
  const file = $("#upload-file").files[0];
  const fd = new FormData();
  fd.append("image", file);
  fd.append("confidence", $("#up-conf").value);
  $("#upload-msg").textContent = "Détection…";
  try {
    const res = await api("/api/detect", { method: "POST", body: fd });
    const img = new Image();
    img.src = URL.createObjectURL(file);
    await img.decode();
    const cv = $("#upload-canvas"), ctx = cv.getContext("2d");
    cv.width = img.width; cv.height = img.height;
    ctx.drawImage(img, 0, 0);
    const lw = Math.max(2, Math.round(Math.max(img.width, img.height) / 500));
    ctx.font = `${Math.max(12, lw * 6)}px system-ui`;
    for (const b of res.boxes) {
      ctx.strokeStyle = "#f97316"; ctx.lineWidth = lw;
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
      const t = `${(b.confidence * 100).toFixed(0)}%`, tw = ctx.measureText(t).width;
      ctx.fillStyle = "#f97316"; ctx.fillRect(b.x1, b.y1 - lw * 8, tw + 8, lw * 8);
      ctx.fillStyle = "#fff"; ctx.fillText(t, b.x1 + 4, b.y1 - lw * 2);
    }
    $("#upload-view").classList.remove("hidden");
    $("#upload-msg").textContent = `${res.boxes.length} navires, ${res.tiles} tuiles, ${(res.seconds * 1000).toFixed(0)} ms sur la 3090.`;
  } catch (err) {
    $("#upload-msg").textContent = `Erreur : ${err.message}`;
  }
};

// ---------- onglets, santé, démarrage ----------

$$("#tabs button").forEach((b) => (b.onclick = () => {
  $$("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
  $$("[data-panel]").forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== b.dataset.tab));
  if (b.dataset.tab === "history") loadHistory();
}));

async function health() {
  try {
    const h = await api("/api/health");
    const hot = h.vram_celsius != null && h.vram_celsius >= h.vram_pause_at - 6;
    $("#health").innerHTML = `
      <span><span class="dot ${h.gpu ? "ok" : "warn"}"></span>${esc(h.gpu?.replace("NVIDIA GeForce ", "") || "CPU")}</span>
      <span>VRAM <b class="${hot ? "warn" : ""}">${h.vram_celsius ?? "?"} °C</b></span>
      <span><span class="dot ${h.vlm_ready ? "ok" : "warn"}"></span>${esc(h.vlm)}</span>
      ${h.jobs_active ? `<span><b>${h.jobs_active}</b> tâche en cours</span>` : ""}`;
  } catch {
    $("#health").textContent = "API injoignable";
  }
}

async function init() {
  const presets = await api("/api/presets");
  $("#presets").innerHTML = presets.map((p) => `<button class="preset" data-id="${p.id}" title="${esc(p.why)}">${esc(p.name)}</button>`).join("");
  $$(".preset").forEach((b) => (b.onclick = () => {
    const p = presets.find((x) => x.id === b.dataset.id);
    setAoi(p.bbox, p.name, p.id);
    map.fitBounds([[p.bbox[0], p.bbox[1]], [p.bbox[2], p.bbox[3]]], { padding: 40, duration: 900 });
  }));
  health();
  setInterval(health, 10000);
  const params = new URLSearchParams(location.search);
  const last = params.get("scan") || (await api("/api/scans"))[0]?.id;
  if (last) {
    await openScan(last);
    const m = state.scan.meta;
    setAoi(m.bbox, m.name, presets.find((p) => p.name === m.name)?.id ?? null);
  }
}
