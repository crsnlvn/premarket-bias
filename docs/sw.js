// App shell is cached for offline open; the report is always fetched fresh first.
const CACHE = 'bias-v55';
const SHELL = ['./', 'index.html', 'fonts/plexmono-400.woff2', 'fonts/plexmono-500.woff2', 'fonts/plexsans-400.woff2', 'fonts/plexsans-600.woff2', 'manifest.webmanifest', 'icons/icon-b-192.png', 'icons/icon-b-180.png'];
self.addEventListener('install', e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL))); self.skipWaiting(); });
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))));
  self.clients.claim();
});
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  if (new URL(e.request.url).origin !== self.location.origin) return;   // news images etc. go straight to the network
  e.respondWith(
    fetch(e.request).then(r => {
      const copy = r.clone();
      caches.open(CACHE).then(c => c.put(e.request.url.split('?')[0], copy));
      return r;
    }).catch(() => caches.match(e.request.url.split('?')[0]).then(r => r || caches.match('index.html')))
  );
});
