/* Aluguel Curitiba — front-end (vanilla JS + Leaflet). */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const BRL = new Intl.NumberFormat("pt-BR", { style: "currency", currency: "BRL", maximumFractionDigits: 0 });
const NUM = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 0 });
const brl = (v) => (v == null ? "—" : BRL.format(v));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const median = (arr) => {
  const a = arr.filter((v) => v != null).sort((x, y) => x - y);
  if (!a.length) return null;
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
};
const norm = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const state = {
  meta: null, listings: [], geo: null,
  mode: "sim",
  point: null,
  form: { area: 150, quartos: 3, banheiros: 2, garagem: true },
  result: null,
  metric: "aluguel",
  filters: { min: null, max: null, rooms: new Set() },
  selectedBairro: null,
};

/* ================================================================ mapa */
const map = L.map("map", { zoomControl: false, minZoom: 10, zoomSnap: 0.25 }).setView([-25.4284, -49.2733], 12);
L.control.zoom({ position: "topright" }).addTo(map);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
}).addTo(map);

let bairrosLayer = null;
let pointsLayer = L.layerGroup();
let similarLayer = L.layerGroup().addTo(map);
let simMarker = null;

const pinIcon = L.divIcon({ className: "", html: '<div class="pin"><span>🏠</span></div>', iconSize: [34, 34], iconAnchor: [17, 38] });
const numIcon = (n, hl) => L.divIcon({ className: "", html: `<div class="num-marker${hl ? " hl" : ""}">${n}</div>`, iconSize: [24, 24], iconAnchor: [12, 12] });

map.on("click", (e) => {
  if (state.mode === "sim") setPoint(e.latlng.lat, e.latlng.lng);
});

/* ================================================================ dados */
async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok && r.status !== 503) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function init() {
  try {
    const [meta, geo, listings] = await Promise.all([getJSON("/api/meta"), getJSON("/api/bairros"), getJSON("/api/anuncios")]);
    Object.assign(state, { meta, geo, listings });
  } catch (err) {
    $("#brand-sub").textContent = "erro ao carregar dados";
    toast("Não consegui carregar a base. Rode `python -m pipeline` e reinicie o servidor.");
    console.error(err);
    return;
  }
  const date = new Date(state.meta.data_coleta + "T12:00").toLocaleDateString("pt-BR", { month: "long", year: "numeric" });
  $("#brand-sub").textContent = `${NUM.format(state.meta.total_anuncios)} casas anunciadas · ${date}`;

  bairrosLayer = L.geoJSON(state.geo, { style: bairroStyle, onEachFeature: bindBairro }).addTo(map);
  map.fitBounds(bairrosLayer.getBounds(), { padding: [10, 10] });
  map.setMaxBounds(bairrosLayer.getBounds().pad(0.3));
  buildAbout();
  readHash();
  renderExplore();
}

/* ================================================================ modo */
$$(".mode").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));

function setMode(mode) {
  state.mode = mode;
  $$(".mode").forEach((b) => { b.classList.toggle("active", b.dataset.mode === mode); b.setAttribute("aria-selected", b.dataset.mode === mode); });
  $("#panel-sim").hidden = mode !== "sim";
  $("#panel-explore").hidden = mode !== "explore";
  $("#map-hint").hidden = mode !== "sim" || !!state.point;
  $("#legend").hidden = mode !== "explore";
  if (mode === "sim") {
    map.removeLayer(pointsLayer);
    similarLayer.addTo(map);
    if (simMarker) simMarker.addTo(map);
  } else {
    map.removeLayer(similarLayer);
    if (simMarker) map.removeLayer(simMarker);
    if ($("#show-points").checked) pointsLayer.addTo(map);
  }
  restyleBairros();
  map.getContainer().style.cursor = mode === "sim" ? "crosshair" : "";
}

/* ================================================================ bairros (polígonos) */
function bairroStyle(feature) {
  const name = feature.properties.bairro;
  if (state.mode === "sim") {
    return { color: "#555", weight: 0.7, opacity: 0.5, fillOpacity: 0 };
  }
  const s = state.stats?.get(name);
  const sel = state.selectedBairro === name;
  const base = { color: sel ? cssVar("--accent") : "#555", weight: sel ? 3 : 0.7, opacity: sel ? 1 : 0.6 };
  if (!s) return { ...base, fillColor: "#999", fillOpacity: 0.08 };
  if (s.count < state.meta.min_anuncios) return { ...base, fillColor: "#aaa", fillOpacity: 0.35, dashArray: "4" };
  return { ...base, fillColor: colorFor(metricValue(s)), fillOpacity: 0.62 };
}

function bindBairro(feature, layer) {
  const name = feature.properties.bairro;
  layer.bindTooltip(() => {
    if (state.mode === "sim") return `<strong>${esc(name)}</strong>`;
    const s = state.stats?.get(name);
    if (!s) return `<strong>${esc(name)}</strong><br>Nenhum anúncio com esses filtros`;
    const few = s.count < state.meta.min_anuncios ? "<br><em>⚠️ poucos anúncios — números instáveis</em>" : "";
    return `<strong>${esc(name)}</strong><br>${s.count} anúncios · mediana ${brl(s.rent)}<br>${brl(s.m2)}/m²${few}`;
  }, { sticky: true, direction: "top", offset: [0, -8] });
  layer.on("click", (e) => {
    if (state.mode !== "explore") return;
    L.DomEvent.stopPropagation(e);
    selectBairro(name);
  });
}

function restyleBairros() {
  if (bairrosLayer) bairrosLayer.setStyle(bairroStyle);
}

/* ================================================================ SIMULADOR */
function setPoint(lat, lon, { fly = false } = {}) {
  state.point = [lat, lon];
  if (!simMarker) {
    simMarker = L.marker([lat, lon], { icon: pinIcon, draggable: true, keyboard: true, title: "Local da casa (arraste para mudar)" }).addTo(map);
    simMarker.on("dragend", () => { const p = simMarker.getLatLng(); setPoint(p.lat, p.lng); });
  } else {
    simMarker.setLatLng([lat, lon]);
  }
  if (fly) map.flyTo([lat, lon], Math.max(map.getZoom(), 14), { duration: 0.6 });
  $("#map-hint").hidden = true;
  requestEstimate();
}

// --- campos
const areaRange = $("#area-range"), areaInput = $("#area");
areaRange.addEventListener("input", () => { areaInput.value = areaRange.value; state.form.area = +areaRange.value; requestEstimate(); });
areaInput.addEventListener("input", () => {
  const v = +areaInput.value;
  if (v >= 20 && v <= 2000) { state.form.area = v; areaRange.value = Math.min(600, Math.max(30, v)); requestEstimate(); }
});
areaInput.addEventListener("blur", () => { areaInput.value = state.form.area; });

$$(".stepper").forEach((el) => {
  const key = el.dataset.key, min = +el.dataset.min, max = +el.dataset.max;
  const out = el.querySelector("output");
  const sync = () => {
    out.textContent = state.form[key];
    el.querySelector('[data-d="-1"]').disabled = state.form[key] <= min;
    el.querySelector('[data-d="1"]').disabled = state.form[key] >= max;
  };
  el.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    state.form[key] = Math.min(max, Math.max(min, state.form[key] + +b.dataset.d));
    sync(); requestEstimate();
  }));
  el.sync = sync;
  sync();
});

segmented($("#garagem"), (v) => { state.form.garagem = v === "1"; requestEstimate(); });

function segmented(root, onChange) {
  root.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    root.querySelectorAll("button").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-checked", x === b); });
    onChange(b.dataset.v);
  }));
}

function syncForm() {
  areaInput.value = state.form.area;
  areaRange.value = Math.min(600, Math.max(30, state.form.area));
  $$(".stepper").forEach((el) => el.sync());
  $$("#garagem button").forEach((b) => b.classList.toggle("on", (b.dataset.v === "1") === state.form.garagem));
}

// --- estimativa (com debounce e descarte de respostas antigas)
let estimateTimer = null, estimateSeq = 0;
function requestEstimate() {
  if (!state.point) return;
  clearTimeout(estimateTimer);
  $("#result-body").classList.add("loading");
  estimateTimer = setTimeout(runEstimate, 220);
}

async function runEstimate() {
  const seq = ++estimateSeq;
  const [lat, lon] = state.point;
  let data;
  try {
    data = await getJSON("/api/estimar", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, ...state.form }),
    });
  } catch (err) {
    if (seq === estimateSeq) toast("Não foi possível calcular agora. Tente de novo.");
    return;
  }
  if (seq !== estimateSeq) return;
  if (data.fora) {
    toast(data.erro);
    $("#place").innerHTML = "📍 Fora de Curitiba — escolha um ponto dentro do município.";
    $("#result-empty").hidden = false;
    $("#result-body").hidden = true;
    similarLayer.clearLayers();
    return;
  }
  state.result = data;
  writeHash();
  renderResult();
}

function confidence(r) {
  if (r.avisos.length >= 2) return ["Confiança baixa", "high"];
  if (r.avisos.length === 1 || r.anuncios_no_bairro < 15) return ["Confiança média", "mid"];
  return ["Confiança boa", "low"];
}

function renderResult() {
  const r = state.result;
  $("#result-empty").hidden = true;
  $("#result-body").hidden = false;
  $("#result-body").classList.remove("loading");
  $("#place").innerHTML = `📍 <strong>${esc(r.bairro)}</strong> · ${r.anuncios_no_bairro} anúncios neste bairro · <span class="muted">arraste o pino para ajustar</span>`;
  $("#r-where").textContent = `Casa de ${NUM.format(state.form.area)} m² em ${r.bairro}`;
  $("#r-value").textContent = brl(r.valor);
  $("#r-low").textContent = brl(r.minimo);
  $("#r-high").textContent = brl(r.maximo);
  // posição do valor dentro da faixa (escala log, mesma lógica do intervalo do modelo)
  const pos = (Math.log(r.valor) - Math.log(r.minimo)) / (Math.log(r.maximo) - Math.log(r.minimo));
  $("#r-dot").style.left = `${12 + 76 * pos}%`;

  const bandClass = { Baixa: "low", "Média": "mid", Alta: "high" }[r.faixa] || "";
  const [lo, hi] = r.faixa_limites;
  const bandText = hi == null ? `acima de ${brl(lo)}` : lo ? `${brl(lo)} a ${brl(hi)}` : `até ${brl(hi)}`;
  const band = $("#r-band");
  band.className = `chip ${bandClass}`;
  band.textContent = `Faixa ${r.faixa.toLowerCase()} (${bandText})`;
  band.title = "Comparado às outras casas da base: as baratas, as do meio e as caras (um terço cada).";
  const [confText, confClass] = confidence(r);
  const conf = $("#r-conf");
  conf.className = `chip ${confClass}`;
  conf.textContent = confText;
  conf.title = `A faixa provável acerta cerca de ${Math.round(state.meta.intervalo_cobertura * 100)}% das vezes nos testes.`;
  $("#r-warnings").innerHTML = r.avisos.map((w) => `<li>${esc(w)}</li>`).join("");

  renderVerdict();
  renderFactors(r.fatores);
  renderSimilar(r.semelhantes);
}

function renderFactors(factors) {
  const maxAbs = Math.max(1, ...factors.map((f) => Math.abs(f.variacao)));
  $("#r-factors").innerHTML = factors.map((f) => {
    const w = (Math.abs(f.variacao) / maxAbs) * 50;
    const dir = f.variacao >= 0 ? "up" : "down";
    const sign = f.variacao >= 0 ? "+" : "−";
    return `<div class="factor"><span>${esc(f.cenario)}</span>
      <div class="track"><div class="bar ${dir}" style="width:${w}%"></div></div>
      <span class="val" style="color:var(--${dir === "up" ? "high" : "low"})">${sign}${BRL.format(Math.abs(f.variacao))}</span></div>`;
  }).join("");
}

function renderSimilar(items) {
  similarLayer.clearLayers();
  const cards = $("#r-similar");
  cards.innerHTML = "";
  items.forEach((s, i) => {
    const n = i + 1;
    const details = [s.area && `${NUM.format(s.area)} m²`, s.quartos && `${s.quartos} qto`, s.banheiros && `${s.banheiros} banh`].filter(Boolean).join(" · ");
    const card = document.createElement("a");
    card.className = "card";
    card.href = s.url;
    card.target = "_blank";
    card.rel = "noopener";
    card.innerHTML = `<span class="n">${n}</span>
      <span><span class="t">${esc(s.bairro)}</span><br><span class="s">${esc(details)} · ${s.distancia_km.toFixed(1).replace(".", ",")} km · ${esc(s.fonte)}</span></span>
      <span class="p">${brl(s.aluguel)}<small>ver anúncio ↗</small></span>`;
    const marker = L.marker([s.latitude, s.longitude], { icon: numIcon(n) })
      .bindTooltip(`<strong>${brl(s.aluguel)}</strong>/mês · ${esc(details)}<br>${esc(s.endereco || s.bairro)}`, { direction: "top", offset: [0, -10] })
      .on("click", () => window.open(s.url, "_blank", "noopener"))
      .addTo(similarLayer);
    card.addEventListener("mouseenter", () => { marker.setIcon(numIcon(n, true)); marker.setZIndexOffset(1000); });
    card.addEventListener("mouseleave", () => { marker.setIcon(numIcon(n)); marker.setZIndexOffset(0); });
    marker.on("mouseover", () => card.classList.add("hl"));
    marker.on("mouseout", () => card.classList.remove("hl"));
    cards.appendChild(card);
  });
}

// --- comparar preço anunciado
$("#announced").addEventListener("input", renderVerdict);
function renderVerdict() {
  const v = +$("#announced").value, r = state.result, el = $("#verdict");
  if (!v || !r) { el.hidden = true; return; }
  const diff = Math.round((v / r.valor - 1) * 100);
  const pct = `${diff > 0 ? "+" : ""}${diff}%`;
  let cls, text;
  if (v > r.maximo) [cls, text] = ["high", `🔺 <strong>Acima do esperado.</strong> ${brl(v)} está ${pct} da estimativa e passa do teto provável (${brl(r.maximo)}). Vale negociar ou entender o que essa casa tem de diferente.`];
  else if (v < r.minimo) [cls, text] = ["low", `🟢 <strong>Abaixo do esperado.</strong> ${brl(v)} está ${pct} da estimativa. Pode ser uma boa oportunidade — confira o estado do imóvel.`];
  else [cls, text] = ["mid", `✅ <strong>Preço dentro do normal</strong> para uma casa assim nessa região (${pct} da estimativa).`];
  el.className = `verdict ${cls}`;
  el.innerHTML = text;
  el.hidden = false;
}

// --- busca de endereço / bairro
const searchInput = $("#search"), searchList = $("#search-results");
let searchTimer = null, searchSeq = 0, searchItems = [], searchActive = -1;

searchInput.addEventListener("input", () => {
  const q = searchInput.value.trim();
  clearTimeout(searchTimer);
  if (q.length < 2) { closeSearch(); return; }
  const nq = norm(q);
  const local = state.meta.bairros.filter((b) => norm(b.nome).includes(nq)).slice(0, 5)
    .map((b) => ({ label: b.nome, hint: "bairro", lat: b.lat, lon: b.lon }));
  showSearch(local, q.length >= 4 ? "Buscando endereços…" : null);
  if (q.length >= 4) searchTimer = setTimeout(() => searchAddress(q, local), 450);
});

async function searchAddress(q, local) {
  const seq = ++searchSeq;
  let hits = [];
  try { hits = await getJSON(`/api/buscar?q=${encodeURIComponent(q)}`); } catch { hits = []; }
  if (seq !== searchSeq || searchInput.value.trim() !== q) return;
  if (!Array.isArray(hits)) { showSearch(local, hits.erro); return; }
  const addr = hits.map((h) => ({ label: h.nome, hint: "endereço", lat: h.lat, lon: h.lon }));
  showSearch([...local, ...addr], local.length + addr.length ? null : "Nada encontrado em Curitiba. Tente rua e número, ou clique no mapa.");
}

function showSearch(items, info) {
  searchItems = items;
  searchActive = -1;
  searchList.innerHTML = items.map((it, i) => `<li role="option" data-i="${i}"><span>${esc(it.label)}</span><small>${it.hint}</small></li>`).join("")
    + (info ? `<li class="info">${esc(info)}</li>` : "");
  searchList.hidden = !items.length && !info;
}
function closeSearch() { searchList.hidden = true; searchItems = []; }
function pickSearch(i) {
  const it = searchItems[i];
  if (!it) return;
  searchInput.value = it.label;
  closeSearch();
  setPoint(it.lat, it.lon, { fly: true });
}
searchList.addEventListener("mousedown", (e) => { const li = e.target.closest("li[data-i]"); if (li) { e.preventDefault(); pickSearch(+li.dataset.i); } });
searchInput.addEventListener("keydown", (e) => {
  if (searchList.hidden || !searchItems.length) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    searchActive = (searchActive + (e.key === "ArrowDown" ? 1 : -1) + searchItems.length) % searchItems.length;
    $$("#search-results li[data-i]").forEach((li) => li.setAttribute("aria-selected", +li.dataset.i === searchActive));
  } else if (e.key === "Enter") {
    e.preventDefault();
    pickSearch(Math.max(0, searchActive));
  } else if (e.key === "Escape") closeSearch();
});
searchInput.addEventListener("blur", () => setTimeout(closeSearch, 150));

// --- link e relatório
function writeHash() {
  const [lat, lon] = state.point, f = state.form;
  history.replaceState(null, "", `#${lat.toFixed(5)},${lon.toFixed(5)},${f.area},${f.quartos},${f.banheiros},${f.garagem ? 1 : 0}`);
}
function readHash() {
  if (location.hash === "#explorar") { setMode("explore"); return; }
  const parts = location.hash.slice(1).split(",").map(Number);
  if (parts.length !== 6 || parts.some((n) => !Number.isFinite(n))) return;
  const [lat, lon, area, quartos, banheiros, garagem] = parts;
  Object.assign(state.form, { area, quartos, banheiros, garagem: !!garagem });
  syncForm();
  setPoint(lat, lon, { fly: true });
}
$("#share-btn").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(location.href); toast("Link copiado — quem abrir verá esta mesma simulação."); }
  catch { toast("Copie o endereço da barra do navegador."); }
});
$("#report-btn").addEventListener("click", () => {
  const r = state.result, f = state.form, [lat, lon] = state.point;
  const lines = [
    "# Estimativa de aluguel — Curitiba", "",
    `Local: ${r.bairro} (${lat.toFixed(5)}, ${lon.toFixed(5)})`,
    `Casa: ${f.area} m², ${f.quartos} quartos, ${f.banheiros} banheiros, ${f.garagem ? "com" : "sem"} garagem`, "",
    `- Aluguel estimado: **${brl(r.valor)}/mês**`,
    `- Faixa provável (${Math.round(state.meta.intervalo_cobertura * 100)}%): ${brl(r.minimo)} a ${brl(r.maximo)}`,
    `- Faixa de preço: ${r.faixa}`,
    ...r.avisos.map((w) => `- Aviso: ${w}`), "",
    "## O que muda o preço", "",
    ...r.fatores.map((x) => `- ${x.cenario}: ${brl(x.valor)} (${x.variacao >= 0 ? "+" : ""}${x.variacao_pct.toFixed(1)}%)`), "",
    "## Casas parecidas", "",
    ...r.semelhantes.map((s) => `- ${s.bairro}, ${s.area ?? "?"} m², ${brl(s.aluguel)} — ${s.url}`), "",
    `Link: ${location.href}`,
  ];
  download("simulacao_aluguel.md", lines.join("\n"), "text/markdown");
});

/* ================================================================ EXPLORAR */
const METRICS = {
  aluguel: { label: "Aluguel mediano", get: (s) => s.rent, fmt: (v) => brl(v) },
  m2: { label: "Preço mediano por m²", get: (s) => s.m2, fmt: (v) => `${brl(v)}/m²` },
  count: { label: "Anúncios", get: (s) => s.count, fmt: (v) => NUM.format(v) },
};
const RAMP = ["#fff3c4", "#fdc877", "#f79b4f", "#e2643b", "#b8332f", "#7a1424"];
const metricValue = (s) => METRICS[state.metric].get(s);
let breaks = [];
function colorFor(v) {
  let i = 0;
  while (i < breaks.length && v > breaks[i]) i++;
  return RAMP[i];
}

segmented($("#metric"), (v) => { state.metric = v; renderExplore(); });
$("#show-points").addEventListener("change", (e) => {
  if (e.target.checked && state.mode === "explore") pointsLayer.addTo(map); else map.removeLayer(pointsLayer);
  renderLegend();
});
$("#f-min").addEventListener("input", (e) => { state.filters.min = e.target.value ? +e.target.value : null; renderExplore(); });
$("#f-max").addEventListener("input", (e) => { state.filters.max = e.target.value ? +e.target.value : null; renderExplore(); });
$$("#f-rooms button").forEach((b) => b.addEventListener("click", () => {
  const v = +b.dataset.v, set = state.filters.rooms;
  set.has(v) ? set.delete(v) : set.add(v);
  b.classList.toggle("on", set.has(v));
  renderExplore();
}));
$("#clear-filters").addEventListener("click", () => {
  state.filters = { min: null, max: null, rooms: new Set() };
  $("#f-min").value = $("#f-max").value = "";
  $$("#f-rooms button").forEach((b) => b.classList.remove("on"));
  renderExplore();
});
$("#rank-filter").addEventListener("input", renderRank);

function filtered() {
  const { min, max, rooms } = state.filters;
  return state.listings.filter((l) =>
    (min == null || l.aluguel >= min) && (max == null || l.aluguel <= max)
    && (!rooms.size || (l.quartos != null && (rooms.has(Math.min(4, l.quartos))))));
}

function bandOf(v) {
  const [a, b] = state.meta.faixas.cortes;
  return v <= a ? "low" : v <= b ? "mid" : "high";
}

function renderExplore() {
  if (!state.meta) return;
  const rows = filtered();
  state.filteredRows = rows;
  // estatísticas por bairro
  const groups = new Map();
  rows.forEach((l) => { if (!groups.has(l.bairro)) groups.set(l.bairro, []); groups.get(l.bairro).push(l); });
  state.stats = new Map([...groups].map(([name, ls]) => [name, {
    count: ls.length, rent: median(ls.map((l) => l.aluguel)), m2: median(ls.map((l) => l.preco_m2)),
  }]));
  // quebras por quantis (só bairros com dados suficientes)
  const vals = [...state.stats.values()].filter((s) => s.count >= state.meta.min_anuncios).map(metricValue).sort((a, b) => a - b);
  breaks = vals.length ? [1, 2, 3, 4, 5].map((k) => vals[Math.min(vals.length - 1, Math.floor((k / 6) * vals.length))]) : [];

  $("#k-count").textContent = NUM.format(rows.length);
  $("#k-rent").textContent = brl(median(rows.map((l) => l.aluguel)));
  $("#k-m2").textContent = rows.length ? brl(median(rows.map((l) => l.preco_m2))) : "—";

  // pontos
  pointsLayer.clearLayers();
  const colors = { low: cssVar("--low"), mid: cssVar("--mid"), high: cssVar("--high") };
  rows.forEach((l) => {
    const info = [l.area && `${NUM.format(l.area)} m²`, l.quartos && `${l.quartos} quartos`].filter(Boolean).join(" · ");
    const est = l.aluguel_estimado != null
      ? `<br>Modelo estimaria ${brl(l.aluguel_estimado)} (${l.desvio_pct > 0 ? "anúncio " + Math.round(l.desvio_pct) + "% acima" : "anúncio " + Math.abs(Math.round(l.desvio_pct)) + "% abaixo"})` : "";
    L.circleMarker([l.latitude, l.longitude], {
      radius: 4, color: "#fff", weight: 0.8, fillColor: colors[bandOf(l.aluguel)], fillOpacity: 0.9,
    }).bindPopup(`<strong>${brl(l.aluguel)}</strong>/mês${info ? " · " + info : ""}<br>${esc(l.endereco || "endereço não informado")} — ${esc(l.bairro)}`
      + `${est}${l.aproximado ? "<br><em>posição aproximada (centro do bairro)</em>" : ""}<br><a href="${esc(l.url)}" target="_blank" rel="noopener">Ver anúncio no ${esc(l.fonte)} ↗</a>`)
      .addTo(pointsLayer);
  });

  restyleBairros();
  renderLegend();
  renderRank();
  if (state.selectedBairro) renderBairroDetail(state.selectedBairro);
}

function renderLegend() {
  const m = METRICS[state.metric];
  const ticks = breaks.length ? [breaks[0], breaks[2], breaks[4]].map((v) => m.fmt(v).replace("/m²", "")) : [];
  $("#legend").innerHTML = `<strong>${m.label} por bairro</strong>
    <div class="scale">${RAMP.map((c) => `<span style="background:${c}"></span>`).join("")}</div>
    <div class="ticks">${ticks.map((t) => `<span>${t}</span>`).join("")}</div>
    <div class="row"><span class="hatch"></span> menos de ${state.meta.min_anuncios} anúncios</div>
    ${$("#show-points").checked ? `<div class="row"><span class="dot" style="background:var(--low)"></span>até ${brl(state.meta.faixas.cortes[0])}
      <span class="dot" style="background:var(--mid)"></span>médio <span class="dot" style="background:var(--high)"></span>acima de ${brl(state.meta.faixas.cortes[1])}</div>` : ""}`;
}

function renderRank() {
  const m = METRICS[state.metric];
  const q = norm($("#rank-filter").value || "");
  $("#rank-label").textContent = state.metric === "count" ? "mais anúncios primeiro" : "mais caros primeiro";
  const items = [...state.stats.entries()]
    .filter(([name]) => !q || norm(name).includes(q))
    .sort(([, a], [, b]) => {
      const fa = a.count >= state.meta.min_anuncios, fb = b.count >= state.meta.min_anuncios;
      return fa !== fb ? (fa ? -1 : 1) : m.get(b) - m.get(a);
    });
  $("#rank").innerHTML = items.map(([name, s]) => {
    const few = s.count < state.meta.min_anuncios;
    return `<li data-b="${esc(name)}" class="${state.selectedBairro === name ? "sel" : ""}">
      <span class="sw" style="background:${few ? "#bbb" : colorFor(m.get(s))}"></span>
      <span>${esc(name)} ${few ? `<span class="few">· ${s.count} anúncio${s.count > 1 ? "s" : ""}</span>` : ""}</span>
      <span class="v">${m.fmt(m.get(s))}</span></li>`;
  }).join("") || '<li class="muted">Nenhum bairro com esses filtros.</li>';
}
$("#rank").addEventListener("click", (e) => { const li = e.target.closest("li[data-b]"); if (li) selectBairro(li.dataset.b, true); });

function selectBairro(name, zoom = false) {
  state.selectedBairro = name;
  restyleBairros();
  renderRank();
  renderBairroDetail(name);
  if (zoom) bairrosLayer.eachLayer((l) => { if (l.feature.properties.bairro === name) map.flyToBounds(l.getBounds(), { padding: [40, 40], duration: 0.6 }); });
  $("#bairro-detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderBairroDetail(name) {
  const s = state.stats.get(name);
  const all = state.listings.filter((l) => l.bairro === name);
  const b = state.meta.bairros.find((x) => x.nome === name);
  const el = $("#bairro-detail");
  el.hidden = false;
  el.innerHTML = `<div class="detail-head"><h2>${esc(name)}</h2><button class="link" id="close-detail">fechar ✕</button></div>
    ${s ? `<div class="detail-grid">
      <div><small>Anúncios</small><strong>${s.count}</strong></div>
      <div><small>Aluguel mediano</small><strong>${brl(s.rent)}</strong></div>
      <div><small>Por m²</small><strong>${brl(s.m2)}</strong></div></div>`
      : `<p class="muted">Nenhum anúncio com os filtros atuais (${all.length} no total).</p>`}
    ${s && s.count < state.meta.min_anuncios ? `<p class="small muted">⚠️ Poucos anúncios: a mediana aqui muda muito com uma casa a mais ou a menos.</p>` : ""}
    <button class="primary" id="sim-here">💰 Simular uma casa neste bairro</button>`;
  $("#close-detail").onclick = () => { state.selectedBairro = null; el.hidden = true; restyleBairros(); renderRank(); };
  $("#sim-here").onclick = () => { setMode("sim"); setPoint(b.lat, b.lon, { fly: true }); };
}

$("#csv-btn").addEventListener("click", () => {
  const cols = ["bairro", "endereco", "aluguel", "area", "quartos", "banheiros", "tem_garagem", "preco_m2", "aluguel_estimado", "desvio_pct", "fonte", "url", "latitude", "longitude"];
  const q = (v) => (v == null ? "" : /[",;\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v);
  const csv = [cols.join(","), ...state.filteredRows.map((r) => cols.map((c) => q(r[c])).join(","))].join("\n");
  download("anuncios_curitiba.csv", csv, "text/csv");
});

/* ================================================================ SOBRE */
function buildAbout() {
  const m = state.meta, mod = m.modelo;
  const imp = Object.entries(mod.importancia);
  const maxImp = Math.max(...imp.map(([, v]) => v));
  let hist = "";
  if (m.historico.length >= 2) {
    const a = m.historico[0], b = m.historico[m.historico.length - 1];
    const d = Math.round((b.aluguel_mediano / a.aluguel_mediano - 1) * 1000) / 10;
    hist = `<h3>Como o preço mudou</h3><p>Entre ${a.data_coleta} e ${b.data_coleta}, o aluguel mediano foi de ${brl(a.aluguel_mediano)} para ${brl(b.aluguel_mediano)} (${d > 0 ? "+" : ""}${String(d).replace(".", ",")}%), comparando só as mesmas fontes. O intervalo é curto: variações pequenas não indicam tendência.</p>`;
  }
  $("#about-body").innerHTML = `
    <h2>Como funciona</h2>
    <p>Reunimos <strong>${NUM.format(m.total_anuncios)} anúncios de casas para alugar</strong> em Curitiba (${m.fontes.join(", ")}). Um modelo de aprendizado de máquina (${esc(mod.regressor)}) aprendeu como área, cômodos, garagem e localização se relacionam com o aluguel pedido.</p>
    <h3>Quão certo ele é?</h3>
    <p>Em casas que o modelo nunca viu, ele erra em média <strong>${brl(mod.erro_medio)}</strong> (cerca de ${Math.round(mod.erro_pct)}%). Por isso mostramos sempre uma <strong>faixa provável</strong>, que contém o valor real em ~${Math.round(m.intervalo_cobertura * 100)}% dos casos.</p>
    <h3>O que mais pesa no preço</h3>
    ${imp.map(([k, v]) => `<div class="imp"><span>${esc(k)}</span><div class="b" style="width:${Math.max(2, (v / maxImp) * 100)}%"></div></div>`).join("")}
    <p class="small muted">Nesta base, tamanho e número de cômodos explicam bem mais do que o bairro. A localização ainda conta, mas menos do que se imagina.</p>
    ${hist}
    <h3>Limitações</h3>
    <ul>
      <li><strong>Preço anunciado não é preço fechado</strong> — na negociação costuma cair.</li>
      <li>O modelo não sabe idade, acabamento, suítes nem estado de conservação.</li>
      <li>Bairros com poucos anúncios (menos de ${m.min_anuncios}) têm estimativas menos confiáveis; o app avisa quando é o caso.</li>
      <li>Parte dos endereços tem posição aproximada.</li>
      <li>Casas de alto padrão aparecem mais nos portais do que na realidade.</li>
    </ul>
    <p class="small muted">Análises técnicas (métricas de todos os modelos, clusters DBSCAN, qualidade dos dados) ficam no painel analítico: <code>./venv/bin/streamlit run app/streamlit_app.py</code></p>`;
}
$("#about-btn").addEventListener("click", () => $("#about").showModal());
$("#about").addEventListener("click", (e) => { if (e.target.id === "about") e.target.close(); });

/* ================================================================ util */
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type: `${type};charset=utf-8` }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 3500);
}

setMode("sim");
init();
