/* Professor de Xadrez — adversário "humano": Maia 3 (CSSLab, Universidade de Toronto; GPL-3.0).
   Rede treinada em milhões de partidas reais: dado o rating, prevê o lance que uma pessoa daquele nível jogaria.
   Entrada: {cmd:"init"}  {cmd:"run", id, tokens(Float32Array 64×12), legal:[uci], elo, opp}
   Saída:   {type:"progress", p}  {type:"ready"}  {type:"result", id, probs}  {type:"error", id?, message} */
importScripts("ort.wasm.min.js");
const BASE = new URL("./", self.location.href).href;
ort.env.wasm.wasmPaths = BASE;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, +(new URLSearchParams(self.location.search).get("t") || 1))) : 1;
const MODEL = "maia3.onnx", MODEL_BYTES = 45683801, CACHE = "px-maia3-v1";
let sess = null, MV = null;

/* baixa o modelo uma vez e guarda no cache do navegador (as próximas partidas abrem na hora) */
async function getModel(){
  const url = BASE + MODEL; let c = null;
  try{ c = await caches.open(CACHE); const hit = await c.match(url); if (hit){ postMessage({ type: "progress", p: 1 }); return await hit.arrayBuffer(); } }catch{ c = null; }
  const r = await fetch(url);
  if (!r.ok) throw new Error(MODEL + ": HTTP " + r.status);
  const rd = r.body.getReader(), parts = []; let got = 0, lastP = 0;
  for (;;){
    const { done, value } = await rd.read(); if (done) break;
    parts.push(value); got += value.length;
    const p = Math.min(0.99, got / MODEL_BYTES); if (p - lastP > 0.01){ lastP = p; postMessage({ type: "progress", p }); }
  }
  const buf = new Uint8Array(got); let o = 0; for (const p of parts){ buf.set(p, o); o += p.length; }
  try{ await c?.put(url, new Response(buf)); }catch{}
  postMessage({ type: "progress", p: 1 });
  return buf.buffer;
}

self.onmessage = async e => {
  const m = e.data;
  try{
    if (m.cmd === "init"){
      const [mv, model] = await Promise.all([fetch(BASE + "all_moves_maia3.json").then(r => r.json()), getModel()]);
      MV = mv;
      sess = await ort.InferenceSession.create(model, { graphOptimizationLevel: "basic" });
      postMessage({ type: "ready" });
    } else if (m.cmd === "run"){
      const out = await sess.run({
        tokens: new ort.Tensor("float32", m.tokens, [1, 64, 12]),
        elo_self: new ort.Tensor("float32", Float32Array.from([m.elo]), [1]),
        elo_oppo: new ort.Tensor("float32", Float32Array.from([m.opp]), [1]),
      });
      const lg = out.logits_move.data, L = m.legal.map(u => MV[u] === undefined ? -1e9 : lg[MV[u]]);
      const mx = Math.max(...L), ex = L.map(v => Math.exp(v - mx)), s = ex.reduce((a, b) => a + b, 0);
      postMessage({ type: "result", id: m.id, probs: ex.map(v => v / s) });
    }
  }catch(err){ postMessage({ type: "error", id: m.id, message: String(err?.message || err) }); }
};
