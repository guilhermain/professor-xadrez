/* Professor de Xadrez — isolamento entre origens (COOP/COEP) no GitHub Pages.
   O GitHub Pages não deixa mudar cabeçalhos; este service worker os acrescenta às respostas
   do próprio site, o que libera SharedArrayBuffer → voz e motor usando vários núcleos.
   Baseado no coi-serviceworker v0.1.7 (Guido Zuidhof e colaboradores, licença MIT).
   Diferença: só mexe em pedidos do próprio site; pedidos para fora (fontes, API do Claude)
   seguem direto, sem passar por aqui. */
if (typeof window === "undefined") {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", e => e.waitUntil(self.clients.claim()));
  self.addEventListener("fetch", e => {
    const r = e.request;
    if (new URL(r.url).origin !== self.location.origin) return;
    if (r.cache === "only-if-cached" && r.mode !== "same-origin") return;
    e.respondWith(fetch(r).then(res => {
      if (res.status === 0) return res;
      const h = new Headers(res.headers);
      h.set("Cross-Origin-Embedder-Policy", "require-corp");
      h.set("Cross-Origin-Opener-Policy", "same-origin");
      h.set("Cross-Origin-Resource-Policy", "same-origin");
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
    }));
  });
} else (() => {
  const n = navigator;
  if (window.crossOriginIsolated !== false || !window.isSecureContext || !n.serviceWorker) return;
  let tries = 0; try{ tries = +sessionStorage.getItem("px_coi") || 0; }catch{}
  if (tries >= 2) return;   // já recarregou e não isolou: segue sem (funciona, só usa 1 núcleo por cópia)
  const reload = () => { try{ sessionStorage.setItem("px_coi", tries + 1); }catch{} location.reload(); };
  n.serviceWorker.register(document.currentScript.src).then(reg => {
    if (reg.active && !n.serviceWorker.controller) reload();
    else if (!n.serviceWorker.controller) n.serviceWorker.addEventListener("controllerchange", reload, { once: true });
  }, () => {});
})();
