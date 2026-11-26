/* Professor de Xadrez — voz neural em segundo plano (worker).
   Kokoro-82M (voz "dora", feminina pt-BR) e Piper (voz "faber", masculina pt-BR),
   ambos via onnxruntime-web; fonemas pelo espeak-ng (piper_phonemize).
   Mensagens de entrada:  {cmd:"load", voice}  {cmd:"synth", id, voice, text, speed, prio}  {cmd:"dropLow"}
   Mensagens de saída:    {type:"progress"|"ready"|"audio"|"error"|"dropped"|"log", ...} */
/* ?gpu=1 → onnxruntime 1.22 com WebGPU (placa de vídeo); senão onnxruntime 1.18 no processador */
const GPU = new URLSearchParams(self.location.search).get("gpu") === "1";
importScripts(GPU ? "ort-gpu.min.js" : "ort.wasm.min.js", "piper_phonemize.js");
const BASE = new URL("./", self.location.href).href;
ort.env.wasm.numThreads = 1;
if (GPU) ort.env.wasm.wasmPaths = { mjs: BASE + "ort-gpu-glue.js", wasm: BASE + "ortgpu.wasm" };
else ort.env.wasm.wasmPaths = BASE;

const VOICES = {
  dora:  { kind: "kokoro", parts: [0,1,2,3,4,5,6].map(i => "kk.onnx." + i + ".wasm"), total: 92361116, style: "kk_pf_dora.bin.wasm", tokenizer: "kk_tokenizer.json", sr: 24000 },   // modelo q8: 0 frases quebradas em 60 (o fp16 quebrou 4)
};
const PP = { parts: ["pp.data.0.wasm", "pp.data.1.wasm"], total: 18077249 };

const log = m => postMessage({ type: "log", message: m });

/* ---------- cache em IndexedDB (sobrevive a recarregar a página) ---------- */
let dbp = null;
function db(){
  if (dbp) return dbp;
  dbp = new Promise(res => {
    try{
      const r = indexedDB.open("px-tts", 1);
      r.onupgradeneeded = () => r.result.createObjectStore("files");
      r.onsuccess = () => res(r.result);
      r.onerror = () => res(null);
    }catch{ res(null); }
  });
  return dbp;
}
async function idbGet(k){ const d = await db(); if (!d) return null; return new Promise(res => { try{ const q = d.transaction("files").objectStore("files").get(k); q.onsuccess = () => res(q.result || null); q.onerror = () => res(null); }catch{ res(null); } }); }
async function idbPut(k, v){ const d = await db(); if (!d) return; return new Promise(res => { try{ const t = d.transaction("files", "readwrite"); t.objectStore("files").put(v, k); t.oncomplete = () => res(); t.onerror = () => res(); }catch{ res(); } }); }

async function fetchBuf(name){
  const r = await fetch(BASE + name);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.arrayBuffer();
}
/* baixa as partes (ou lê do cache) e junta num único buffer */
async function fetchParts(parts, total, onProgress){
  const out = new Uint8Array(total); let off = 0, fromCache = 0;
  for (const p of parts){
    let buf = await idbGet(p);
    if (buf) fromCache++;
    else { buf = await fetchBuf(p); idbPut(p, buf); }
    out.set(new Uint8Array(buf), off); off += buf.byteLength;
    onProgress && onProgress(off, total);
  }
  if (off !== total) throw new Error(`tamanho inesperado: ${off} de ${total}`);
  return { bytes: out, fromCache };
}

/* ---------- fonemas (espeak-ng em WASM) ---------- */
let PH = null;
async function phonemizer(onProgress){
  if (PH) return PH;
  const wasm = new Uint8Array(await fetchBuf("piper_phonemize.wasm"));
  const { bytes } = await fetchParts(PP.parts, PP.total, onProgress);
  PH = { wasm, data: bytes.buffer };
  return PH;
}
async function phonemize(text){
  const ph = await phonemizer();
  return await new Promise(async (resolve, reject) => {
    try{
      const m = await createPiperPhonemize({ wasmBinary: ph.wasm, getPreloadedPackage: () => ph.data,
        print: d => { try{ resolve(JSON.parse(d)) }catch(e){ reject(e) } }, printErr: e => reject(new Error(String(e))) });
      m.callMain(["-l", "pt-br", "--input", JSON.stringify([{ text }]), "--espeak_data", "/espeak-ng-data"]);
    }catch(e){ reject(e); }
  });
}

/* ---------- vozes ---------- */
const LOADED = {};
/* limpa do cache o modelo fp16 (quebrava frases) e o Faber */
(async () => { try{ const d = await db(); if (!d) return; const st = d.transaction("files", "readwrite").objectStore("files");
  const q = st.getAllKeys(); q.onsuccess = () => { for (const k of q.result) if (/^(kkf16\.onnx|ortgpu|faber\.onnx)/.test(k)) st.delete(k); }; }catch{} })();
let GPU_READY = null;
async function gpuRuntime(onProgress){
  if (GPU_READY) return GPU_READY;
  GPU_READY = (async () => {
    const { bytes } = await fetchParts(["ortgpu.0.wasm", "ortgpu.1.wasm"], 21872216, onProgress);
    ort.env.wasm.wasmBinary = bytes.buffer;
  })();
  return GPU_READY;
}
async function load(name, announce){
  if (LOADED[name]){ if (announce) postMessage({ type: "ready", voice: name, ms: 0, fromCache: "memória" }); return LOADED[name]; }
  const v = VOICES[name]; if (!v) throw new Error("voz desconhecida: " + name);
  const t0 = performance.now();
  const grand = v.total + PP.total; let doneModel = 0, donePP = 0;
  const prog = () => postMessage({ type: "progress", voice: name, loaded: doneModel + donePP, total: grand });
  await phonemizer((o) => { donePP = o; prog(); });
  const { bytes, fromCache } = await fetchParts(v.parts, v.total, (o) => { doneModel = o; prog(); });
  if (GPU) await gpuRuntime();
  const sess = await ort.InferenceSession.create(bytes, { executionProviders: [GPU ? "webgpu" : "wasm"] });
  const L = { v, sess };
  if (v.kind === "kokoro"){
    L.vocab = (await (await fetch(BASE + v.tokenizer)).json()).model.vocab;
    L.style = new Float32Array(await fetchBuf(v.style));
    L.sr = v.sr;
  } else {
    L.cfg = await (await fetch(BASE + v.config)).json();
    L.sr = L.cfg.audio.sample_rate;
  }
  if (GPU){   // confere de verdade: gera uma frase curta e testa se o áudio presta (sem NaN, com som)
    const { pcm, headPeak } = await synth(name, "Olá.", 1, L);
    postMessage({ type: "log", message: `teste da placa: ruído no começo (antes da limpeza) ${headPeak.toFixed(3)}` });
    let bad = 0, peak = 0; for (const x of pcm){ if (x !== x) bad++; else if (Math.abs(x) > peak) peak = Math.abs(x); }
    if (bad || peak < 0.01) throw new Error(`teste da placa de vídeo falhou (NaN ${bad}, pico ${peak.toFixed(3)})`);
  }
  LOADED[name] = L;
  postMessage({ type: "ready", voice: name, ms: Math.round(performance.now() - t0), fromCache, gpu: GPU });
  return L;
}

async function synth(name, text, speed, Lpre){
  const L = Lpre || await load(name);
  const ph = await phonemize(text);
  if (L.v.kind === "kokoro"){
    const ipa = (ph.phonemes || []).flat().join("").replace(/([.!?,;:])(?=\S)/g, "$1 ");
    const ids = [0];
    for (const ch of ipa) if (ch in L.vocab) ids.push(L.vocab[ch]);
    if (ids.length > 511) ids.length = 511;
    ids.push(0);
    const n = Math.min(ids.length - 2, 509);
    const style = L.style.slice(n * 256, n * 256 + 256);
    const out = await L.sess.run({
      input_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
      style: new ort.Tensor("float32", style, [1, 256]),
      speed: new ort.Tensor("float32", [speed], [1]),
    });
    return cleanEdges(out.waveform.data, L.sr);
  }
  const inf = L.cfg.inference, ids = ph.phoneme_ids;
  const feeds = {
    input: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    input_lengths: new ort.Tensor("int64", BigInt64Array.from([BigInt(ids.length)])),
    scales: new ort.Tensor("float32", [inf.noise_scale, inf.length_scale / speed, inf.noise_w]),
  };
  if (L.cfg.num_speakers > 1) feeds.sid = new ort.Tensor("int64", BigInt64Array.from([0n]));
  const out = await L.sess.run(feeds);
  return { pcm: out.output.data, sr: L.sr };
}

/* bordas da frase: a fala certa sempre começa depois de ~190 ms de silêncio (medido).
   Zera os primeiros 150 ms (some com o chiado que a placa de vídeo gera ali), e corta o silêncio
   que sobra, deixando 60 ms: a fala começa antes e fica mais fluida. */
function cleanEdges(raw, sr){
  const y = Float32Array.from(raw, x => (x === x && Math.abs(x) <= 1.5) ? x : 0), gate = Math.round(sr * 0.15), fade = Math.round(sr * 0.03);
  let bad = 0; for (const x of raw) if (!(x === x) || Math.abs(x) > 1.5) bad++;
  let headPeak = 0; for (let i = 0; i < Math.min(gate, y.length); i++) headPeak = Math.max(headPeak, Math.abs(y[i]));
  let tailPeak = 0; for (let i = Math.max(0, y.length - gate); i < y.length; i++) tailPeak = Math.max(tailPeak, Math.abs(y[i]));
  for (let i = 0; i < Math.min(gate, y.length); i++) y[i] = 0;
  for (let i = 0; i < fade && gate + i < y.length; i++) y[gate + i] *= i / fade;
  const tail = Math.round(sr * 0.1);                       // fim: 100 ms com fade (também sem ruído)
  for (let i = 0; i < tail && i < y.length; i++) y[y.length - 1 - i] *= i / tail;
  let on = gate; while (on < y.length && Math.abs(y[on]) < 0.01) on++;
  const start = Math.max(0, on - Math.round(sr * 0.06));
  if (bad > 0) return { pcm: new Float32Array(Math.round(sr * 0.05)), sr, headPeak: 0, tailPeak: 0, bad };   // frase quebrada: silêncio, nunca chiado
  return { pcm: y.subarray(start), sr, headPeak, tailPeak, bad };
}
/* ---------- fila: pedidos de fala (prio) passam na frente das pré-gerações ---------- */
const Q = []; let busy = false;
async function pump(){
  if (busy || !Q.length) return;
  busy = true;
  const job = Q.shift();
  if (self.DEBUG) postMessage({ type: "log", message: `início ${job.id || job.cmd} prio=${!!job.prio} fila=${Q.length} "${(job.text || "").slice(0, 30)}"` });
  try{
    if (job.cmd === "load"){ await load(job.voice, true); }
    else {
      const t0 = performance.now();
      const { pcm, sr, headPeak, tailPeak, bad } = await synth(job.voice, job.text, job.speed);
      const f = new Float32Array(pcm);
      postMessage({ type: "audio", id: job.id, pcm: f, sr, ms: Math.round(performance.now() - t0), sec: f.length / sr, headPeak, tailPeak, bad, gpu: GPU }, [f.buffer]);
    }
  }catch(e){
    postMessage({ type: "error", id: job.id, voice: job.voice, cmd: job.cmd, message: String(e && e.message || e) });
  }
  busy = false;
  pump();
}
onmessage = (e) => {
  const m = e.data;
  if (m.cmd === "debug"){ self.DEBUG = true; return; }
  if (m.cmd === "dropLow"){
    const keep = [], dropped = [];
    for (const j of Q) (j.cmd === "synth" && !j.prio ? dropped : keep).push(j);
    Q.length = 0; Q.push(...keep);
    if (dropped.length) postMessage({ type: "dropped", ids: dropped.map(j => j.id) });
    return;
  }
  if (m.cmd === "retain"){
    const keep = new Map(m.keep.map((id, i) => [id, i])), kept = [], dropped = [], other = [];
    const low = [];
    for (const j of Q){ if (j.cmd !== "synth"){ other.push(j); continue; } if (keep.has(j.id)) kept.push(j); else if (j.prio) dropped.push(j); else low.push(j); }
    kept.sort((a, b) => keep.get(a.id) - keep.get(b.id)); kept.forEach(j => j.prio = true);
    Q.length = 0; Q.push(...other, ...kept, ...low);            // a pré-geração de fundo continua depois
    if (dropped.length) postMessage({ type: "dropped", ids: dropped.map(j => j.id) });
    return;
  }
  if (m.cmd === "synth" && m.prio){
    const i = Q.findIndex(j => j.cmd === "synth" && !j.prio);
    if (i < 0) Q.push(m); else Q.splice(i, 0, m);
  } else Q.push(m);
  pump();
};
