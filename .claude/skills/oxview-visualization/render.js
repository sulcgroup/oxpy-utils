#!/usr/bin/env node
/**
 * Render files in oxView (headless Chrome) and save a PNG.
 *
 *   node render.js --out s.png init.top last_conf.dat particles.txt patches.txt
 *
 * oxView is a static app (index.html + the committed dist/), so this just serves the checkout on a
 * random port, starts a throwaway headless Chrome, puts the files into oxView's own file input
 * exactly as a drag-and-drop would, optionally runs scripting-console snippets, screenshots, and
 * cleans up. No build step. The checkout is not modified.
 *
 * Options
 *   --out FILE        PNG to write (default oxview.png)
 *   --repo DIR        oxdna-viewer checkout (default $OXVIEW_DIR, else ~/git/oxdna-viewer-navsikand)
 *   --eval JS         run JS in the page after the files load (repeatable, in order); a returned value is
 *                     printed as JSON, e.g. --eval "api.countStrandLength()" or
 *                     --eval "api.toggleBaseColors()"   (the Scene API and edit API are globals)
 *   --size WxH        viewport (default 1400x900)
 *   --settle MS       wait before the shot (default 2000)
 *   --frame first|last|N   trajectory frame to show (0-based; default: frame 0). Uses the
 *                     loaded system's reader (nextConfig/previousConfig), as oxView's own buttons do
 *   --fit             move the camera to frame everything loaded (oxView keeps its default
 *                     camera, so small systems otherwise render tiny)
 *   --stats           print what oxView loaded (systems, strands, elements)
 *   --chrome PATH     Chrome binary (default $CHROME_PATH, else google-chrome/chromium)
 *
 * Everything else on the command line is a file to load (any order; oxView decides what each is by
 * extension and name: see the skill). Needs `ws`: see the skill for a one-line install that does not
 * touch any repo.
 */
const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');
const net = require('net');
const { spawn } = require('child_process');

function parseArgs(argv) {
  const o = { out: 'oxview.png', size: '1400x900', settle: 2000, files: [], evals: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const val = () => argv[++i];
    if (a === '--out') o.out = val();
    else if (a === '--repo') o.repo = val();
    else if (a === '--eval') o.evals.push(val());
    else if (a === '--size') o.size = val();
    else if (a === '--settle') o.settle = Number(val());
    else if (a === '--stats') o.stats = true;
    else if (a === '--fit') o.fit = true;
    else if (a === '--frame') o.frame = val();
    else if (a === '--chrome') o.chrome = val();
    else if (a.startsWith('--')) throw new Error(`unknown option ${a}`);
    else o.files.push(path.resolve(a));
  }
  return o;
}

const freePort = () => new Promise((resolve, reject) => {
  const s = net.createServer();
  s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => resolve(p)); });
  s.on('error', reject);
});
const sleep = ms => new Promise(r => setTimeout(r, ms));

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon', '.woff2': 'font/woff2',
  '.woff': 'font/woff', '.ttf': 'font/ttf', '.map': 'application/json',
};
function serve(dir, port) {
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(req.url.split('?')[0]);
    if (p === '/') p = '/index.html';
    const file = path.join(dir, p);
    if (!file.startsWith(dir) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404); return res.end();
    }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise(r => server.listen(port, '127.0.0.1', () => r(server)));
}

function findWs(repo) {
  const cache = process.env.OXPY_UTILS_RENDER_DEPS || path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache'), 'oxpy-utils-render-deps');
  for (const dir of [process.env.WS_DIR, path.join(repo, 'node_modules'), path.join(cache, 'node_modules')].filter(Boolean)) {
    try { return require(require.resolve('ws', { paths: [dir, path.dirname(dir)] })); } catch (_) { /* next */ }
  }
  throw new Error(`ws not found. Install it outside any repo: npm install --prefix ${cache} ws`);
}

async function cdpConnect(WebSocket, port) {
  const targets = await new Promise((ok, no) => http.get({ host: '127.0.0.1', port, path: '/json/list' }, r => {
    let b = ''; r.on('data', c => (b += c)); r.on('end', () => ok(JSON.parse(b)));
  }).on('error', no));
  const page = targets.find(t => t.type === 'page');
  if (!page) throw new Error('no page target');
  const ws = new WebSocket(page.webSocketDebuggerUrl, { perMessageDeflate: false });
  let id = 0;
  const pending = new Map();
  const logs = [];
  ws.on('message', raw => {
    const m = JSON.parse(raw);
    if (m.id && pending.has(m.id)) {
      const { resolve, reject } = pending.get(m.id); pending.delete(m.id);
      m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result);
    } else if (m.method === 'Runtime.exceptionThrown') {
      logs.push({ level: 'exception', text: m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text });
    } else if (m.method === 'Runtime.consoleAPICalled' && ['error'].includes(m.params.type)) {
      logs.push({ level: m.params.type, text: (m.params.args || []).map(a => a.value ?? a.description ?? '').join(' ') });
    }
  });
  await new Promise(r => ws.on('open', r));
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const n = ++id;
    const t = setTimeout(() => { pending.delete(n); reject(new Error(`${method} timed out`)); }, 120000);
    pending.set(n, { resolve: v => { clearTimeout(t); resolve(v); }, reject: e => { clearTimeout(t); reject(e); } });
    ws.send(JSON.stringify({ id: n, method, params }));
  });
  await send('Page.enable'); await send('Runtime.enable'); await send('DOM.enable');
  const evaluate = async expression => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    return r.result.value;
  };
  return {
    send, evaluate, logs, close: () => ws.close(),
    async setFiles(selector, files) {
      const doc = await send('DOM.getDocument');
      const node = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector });
      if (!node.nodeId) throw new Error(`no element ${selector}`);
      await send('DOM.setFileInputFiles', { nodeId: node.nodeId, files });
    },
    async screenshot(file) {
      const s = await send('Page.captureScreenshot', { format: 'png' });
      fs.writeFileSync(file, Buffer.from(s.data, 'base64'));
    },
  };
}

(async () => {
  const o = parseArgs(process.argv.slice(2));
  const repo = path.resolve(o.repo || process.env.OXVIEW_DIR || path.join(os.homedir(), 'git/oxdna-viewer-navsikand'));
  if (!fs.existsSync(path.join(repo, 'index.html')) || !fs.existsSync(path.join(repo, 'dist/main.js'))) {
    throw new Error(`${repo} has no index.html + dist/main.js - is --repo right?`);
  }
  if (!o.files.length) throw new Error('no files given');
  for (const f of o.files) if (!fs.existsSync(f)) throw new Error(`no such file: ${f}`);
  const WebSocket = findWs(repo);

  const chrome = o.chrome || process.env.CHROME_PATH ||
    ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
     '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(p => fs.existsSync(p));
  if (!chrome) throw new Error('no Chrome found; pass --chrome or set CHROME_PATH');

  const [width, height] = o.size.split('x').map(Number);
  const webPort = await freePort();
  const debugPort = await freePort();
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'oxview-render-'));
  const server = await serve(repo, webPort);
  const chromeProc = spawn(chrome, [
    '--headless=new', '--no-sandbox', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
    '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', 'about:blank',
  ], { stdio: 'ignore' });
  const cleanup = () => {
    try { chromeProc.kill('SIGTERM'); } catch (_) {}
    try { server.close(); } catch (_) {}
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch (_) {}
  };
  process.on('exit', cleanup);

  try {
    for (let i = 0; ; i++) {
      try { await new Promise((ok, no) => http.get({ host: '127.0.0.1', port: debugPort, path: '/json/version' }, r => { r.resume(); ok(); }).on('error', no)); break; }
      catch (e) { if (i > 100) throw new Error('Chrome did not start'); await sleep(200); }
    }
    const page = await cdpConnect(WebSocket, debugPort);
    await page.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
    await page.send('Page.navigate', { url: `http://127.0.0.1:${webPort}/index.html` });
    for (let i = 0; ; i++) {
      if (await page.evaluate(`!!document.getElementById('fileInput') && typeof handleFiles === 'function'`).catch(() => false)) break;
      if (i > 150) throw new Error('oxView did not finish loading');
      await sleep(200);
    }
    await page.setFiles('#fileInput', o.files);
    // the input lives in the Open dialog, whose Open button does exactly this
    await page.evaluate(`handleFiles(document.getElementById('fileInput').files)`);
    // wait for a system to appear (loading is asynchronous)
    let loaded = false;
    for (let i = 0; i < 300 && !loaded; i++) {
      loaded = await page.evaluate(`typeof systems !== 'undefined' && systems.length > 0`).catch(() => false);
      if (!loaded) await sleep(200);
    }
    if (!loaded) {
      // leave evidence: oxView reports problems as on-page toasts and console errors
      await page.screenshot(path.resolve(o.out));
      console.error(`page errors: ${JSON.stringify(page.logs.slice(0, 8))}`);
      const toasts = await page.evaluate(`[...document.querySelectorAll('.toast, .notify, .info-box')].map(e => e.innerText.trim()).filter(Boolean)`).catch(() => []);
      if (toasts.length) console.error(`on-page messages: ${JSON.stringify(toasts)}`);
      throw new Error(`oxView loaded no system from these files; screenshot of the page written to ${path.resolve(o.out)}`);
    }
    await sleep(500);
    if (o.frame !== undefined) {
      const n = await page.evaluate(`systems[systems.length - 1].reader.lookupReader.position_lookup.length`);
      const target = o.frame === 'first' ? 0 : o.frame === 'last' ? n - 1 : Number(o.frame);
      if (!Number.isInteger(target) || target < 0 || target >= Math.max(n, 1)) throw new Error(`--frame must be first, last or 0..${Math.max(n, 1) - 1} (this file has ${n} frame(s))`);
      for (let guard = 0; guard < 100000; guard++) {
        const idx = await page.evaluate(`systems[systems.length - 1].reader.idx`);
        if (idx === target) break;
        const before = idx;
        await page.evaluate(idx < target ? `systems[systems.length - 1].reader.nextConfig()` : `systems[systems.length - 1].reader.previousConfig()`);
        // reading a frame is asynchronous: wait for the index to move
        for (let i = 0; i < 200 && (await page.evaluate(`systems[systems.length - 1].reader.idx`)) === before; i++) await sleep(50);
      }
      console.log(`frame: ${await page.evaluate(`systems[systems.length - 1].reader.idx`)} of ${n}`);
    }
    if (o.fit) {
      const info = await page.evaluate(`(() => {
        const pts = [];
        elements.forEach(e => { try { pts.push(e.getPos()); } catch (_) {} });
        const box = new THREE.Box3().setFromPoints(pts);
        const c = box.getCenter(new THREE.Vector3());
        const r = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1);
        const dist = (r * 1.15) / Math.tan((camera.fov * Math.PI / 180) / 2);
        const dir = camera.position.clone().sub(controls.target).normalize();
        if (!isFinite(dir.length()) || dir.length() === 0) dir.set(0, 0, 1);
        camera.position.copy(c).add(dir.multiplyScalar(dist));
        controls.target.copy(c);
        camera.lookAt(c);
        camera.updateProjectionMatrix();
        controls.update();
        return { points: pts.length, radius: +r.toFixed(2), distance: +dist.toFixed(2) };
      })()`);
      console.log('fit:', JSON.stringify(info));
    }
    for (const js of o.evals) {
      // print what the snippet returns, so the Scene/Edit API can be used to read things back
      try {
        const v = await page.evaluate(`(() => { const r = (${'0,'} eval(${JSON.stringify(js)})); return r === undefined ? undefined : JSON.stringify(r); })()`);
        if (v !== undefined) console.log(`eval ${JSON.stringify(js)} ->`, v.length > 1500 ? v.slice(0, 1500) + ' ...(truncated)' : v);
      } catch (e) { console.error(`eval ${JSON.stringify(js)} failed: ${String(e.message).split('\n')[0]}`); }
    }
    if (o.stats) {
      console.log('stats:', JSON.stringify(await page.evaluate(`({
        systems: systems.length,
        kinds: systems.map(s => s.constructor.name),
        strands: systems.map(s => s.strands ? s.strands.length : null),
        elements: typeof elements !== 'undefined' && elements.size !== undefined ? elements.size : null,
      })`)));
    }
    await sleep(o.settle);
    await page.screenshot(path.resolve(o.out));
    console.log(`wrote ${path.resolve(o.out)}`);
    // oxView always throws two harmless errors on load (tacoxdna worker loaded as a plain script, and
    // a Metro dialog-close handler); report only the others
    const real = page.logs.filter(l => !/importScripts is not defined|metro\.min\.js/.test(l.text));
    if (real.length) console.log('page errors:', JSON.stringify(real.slice(0, 5)));
    page.close();
  } finally {
    cleanup();
  }
  process.exit(0);
})().catch(e => { console.error(`render.js: ${e.message}`); process.exit(1); });
