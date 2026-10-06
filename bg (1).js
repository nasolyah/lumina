/* Lumina · динамичный фон «маркер по бумаге».
 * Рисует на <canvas id="bg-canvas">: каракули маркера на полях тетради (кольца, волны, галочки,
 * подчёркивания, плюсики) + «карту мыслей» из узлов. Клетка-точка — CSS на самом канвасе.
 * Цвета берутся ТОЛЬКО из CSS-переменных (--br-1…--br-6, --accent, --ink) через getComputedStyle.
 *
 * API:  window.LuminaBg.init({ intensity, speed, nodes, doodles })   window.LuminaBg.destroy()
 * Автозапуск на странице, если у <script> нет атрибута data-manual (app.html запускает вручную).
 */
(function () {
  'use strict';

  var AUTO = !(document.currentScript && document.currentScript.hasAttribute('data-manual'));

  /* ── НАСТРОЙКИ (меняйте здесь) ── */
  var CONFIG = {
    intensity: 1,            // общая «громкость»: множитель прозрачности и скорости (0.5 = тише)
    speed: 1,                // множитель скорости дрейфа всего фона
    nodes: [15, 25],         // число узлов (мин, макс); на ширине < 700px — вдвое меньше
    doodles: [9, 12],        // число каракулей маркера (мин, макс); на ширине < 700px — вдвое меньше
    nodeSpeed: [6, 14],      // px/сек, скорость узлов
    doodleSpeed: [5, 15],    // px/сек, скорость дрейфа каракулей
    doodleLife: [24, 44],    // сек, жизнь каракули (плавно появляется и исчезает)
    doodleSize: [26, 52],    // px
    doodleSpin: 0.12,        // макс. скорость вращения, рад/сек
    doodleAlphaLight: [0.6, 0.9],
    doodleAlphaDark: [0.35, 0.6],
    doodleWidth: 3.5,
    nodeAlpha: 0.25,
    nodeRadius: [3, 6],
    linkDist: 140,           // узлы ближе этого — соединяются линией
    linkAlpha: 0.12,         // макс. прозрачность линии (при нулевом расстоянии)
    mouseRadius: 120,        // радиус отталкивания от курсора
    mouseForce: 260,         // сила отталкивания, px/сек²
    mobileWidth: 700
  };

  var SHAPES = ['ring', 'squiggle', 'check', 'underline', 'plus'];
  var BRANCH_VARS = ['--br-1', '--br-2', '--br-3', '--br-4', '--br-5', '--br-6'];
  // запасные имена токенов (в app.html чернила называются --text)
  var INK_VARS = ['--ink', '--text'];

  var mqDark = window.matchMedia('(prefers-color-scheme: dark)');
  var mqMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  var mqHover = window.matchMedia('(hover: hover) and (pointer: fine)');

  var S = null; // состояние активного экземпляра

  function rnd(a, b) { return a + Math.random() * (b - a); }
  function rndInt(a, b) { return Math.floor(rnd(a, b + 1)); }

  function cssVar(names) {
    var cs = getComputedStyle(document.documentElement);
    for (var i = 0; i < names.length; i++) {
      var v = cs.getPropertyValue(names[i]).trim();
      if (v) return v;
    }
    return '';
  }

  function isDark() {
    var t = document.documentElement.getAttribute('data-theme');
    if (t === 'dark') return true;
    if (t === 'light') return false;
    return mqDark.matches;
  }

  function readColors() {
    S.dark = isDark();
    S.colors = BRANCH_VARS.map(function (n) { return cssVar([n]); }).filter(Boolean);
    S.accent = cssVar(['--accent']);
    S.ink = cssVar(INK_VARS);
    if (S.reduced) draw();
  }

  /* ── создание объектов ── */
  function newDoodle(initial) {
    var life = rnd(CONFIG.doodleLife[0], CONFIG.doodleLife[1]);
    var dir = rnd(0, Math.PI * 2);                    // направление дрейфа
    var sp = rnd(CONFIG.doodleSpeed[0], CONFIG.doodleSpeed[1]);
    return {
      type: SHAPES[Math.floor(Math.random() * SHAPES.length)],
      x: rnd(0, S.w), y: rnd(0, S.h),
      size: rnd(CONFIG.doodleSize[0], CONFIG.doodleSize[1]),
      rot: rnd(-Math.PI, Math.PI), spin: rnd(-CONFIG.doodleSpin, CONFIG.doodleSpin),
      vx: Math.cos(dir) * sp, vy: Math.sin(dir) * sp,
      ci: Math.floor(Math.random() * 6),
      u: Math.random(),                               // позиция внутри диапазона прозрачности
      life: life,
      age: initial ? rnd(0, life) : 0
    };
  }

  // контуры каракулей в локальных координатах (центр в 0,0); s — размер
  function shapePath(ctx, type, s) {
    var i, x;
    ctx.beginPath();
    if (type === 'ring') {                            // неровное «обведено»: две дуги со смещением
      ctx.arc(0, 0, s * 0.5, 0.4, 0.4 + Math.PI * 1.8);
      ctx.moveTo(s * 0.46 * Math.cos(5.6), s * 0.46 * Math.sin(5.6));
      ctx.arc(s * 0.02, s * 0.02, s * 0.46, 5.6, 5.6 + Math.PI * 0.75);
    } else if (type === 'squiggle') {
      for (i = 0; i <= 24; i++) {
        x = -s + (i / 24) * 2 * s;
        var y = Math.sin((i / 24) * Math.PI * 5) * s * 0.2;
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      }
    } else if (type === 'check') {
      ctx.moveTo(-s * 0.45, s * 0.02); ctx.lineTo(-s * 0.12, s * 0.34); ctx.lineTo(s * 0.5, -s * 0.38);
    } else if (type === 'underline') {
      ctx.moveTo(-s * 0.8, 0); ctx.quadraticCurveTo(0, -s * 0.12, s * 0.8, s * 0.02);
      ctx.moveTo(-s * 0.5, s * 0.24); ctx.quadraticCurveTo(0, s * 0.14, s * 0.55, s * 0.26);
    } else {                                          // plus
      ctx.moveTo(-s * 0.35, 0); ctx.lineTo(s * 0.35, 0.02 * s);
      ctx.moveTo(0.02 * s, -s * 0.35); ctx.lineTo(0, s * 0.35);
    }
  }

  function newNode() {
    var dir = rnd(0, Math.PI * 2);
    var sp = rnd(CONFIG.nodeSpeed[0], CONFIG.nodeSpeed[1]);
    return {
      x: rnd(0, S.w), y: rnd(0, S.h),
      vx: Math.cos(dir) * sp, vy: Math.sin(dir) * sp,
      px: 0, py: 0,                                   // импульс от мыши (затухает)
      r: rnd(CONFIG.nodeRadius[0], CONFIG.nodeRadius[1])
    };
  }

  function targetCounts() {
    var k = S.w < CONFIG.mobileWidth ? 0.5 : 1;
    return {
      nodes: Math.max(4, Math.round(S.baseNodes * k)),
      doodles: Math.max(2, Math.round(S.baseDoodles * k))
    };
  }

  function fitCounts() {
    var t = targetCounts(), i;
    while (S.nodes.length < t.nodes) S.nodes.push(newNode());
    S.nodes.length = t.nodes;
    while (S.doodles.length < t.doodles) S.doodles.push(newDoodle(true));
    S.doodles.length = t.doodles;
  }

  /* ── размеры ── */
  function resize() {
    var r = S.canvas.getBoundingClientRect();
    var w = Math.max(1, r.width), h = Math.max(1, r.height);
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var kx = S.w ? w / S.w : 1, ky = S.h ? h / S.h : 1;
    S.nodes.concat(S.doodles).forEach(function (o) { o.x *= kx; o.y *= ky; });
    S.w = w; S.h = h; S.dpr = dpr;
    S.canvas.width = Math.round(w * dpr);
    S.canvas.height = Math.round(h * dpr);
    S.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fitCounts();
    if (S.reduced) draw();
  }

  function onResize() {
    clearTimeout(S.resizeTimer);
    S.resizeTimer = setTimeout(function () { if (S) resize(); }, 150);
  }

  /* ── отрисовка ── */
  function draw() {
    var ctx = S.ctx, i, j, a, b;
    var k = CONFIG.intensity;
    ctx.clearRect(0, 0, S.w, S.h);
    if (!S.colors.length) return;

    // каракули маркера
    var range = S.dark ? CONFIG.doodleAlphaDark : CONFIG.doodleAlphaLight;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = CONFIG.doodleWidth;
    for (i = 0; i < S.doodles.length; i++) {
      var s = S.doodles[i];
      var env = Math.sin(Math.PI * Math.min(1, s.age / s.life));   // 0 → 1 → 0
      var alpha = (range[0] + s.u * (range[1] - range[0])) * env * k;
      if (alpha <= 0.001) continue;
      ctx.save();
      ctx.translate(s.x, s.y);
      ctx.rotate(s.rot);
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = S.colors[s.ci % S.colors.length];
      shapePath(ctx, s.type, s.size);
      ctx.stroke();
      ctx.restore();
    }

    // связи между близкими узлами
    if (S.ink) {
      ctx.strokeStyle = S.ink;
      ctx.lineWidth = 1;
      ctx.lineCap = 'butt';
      for (i = 0; i < S.nodes.length; i++) {
        a = S.nodes[i];
        for (j = i + 1; j < S.nodes.length; j++) {
          b = S.nodes[j];
          var dx = a.x - b.x, dy = a.y - b.y;
          var d = Math.sqrt(dx * dx + dy * dy);
          if (d >= CONFIG.linkDist) continue;
          ctx.globalAlpha = CONFIG.linkAlpha * (1 - d / CONFIG.linkDist) * k;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    // узлы
    if (S.accent) {
      ctx.fillStyle = S.accent;
      ctx.globalAlpha = CONFIG.nodeAlpha * k;
      for (i = 0; i < S.nodes.length; i++) {
        a = S.nodes[i];
        ctx.beginPath();
        ctx.arc(a.x, a.y, a.r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  /* ── физика ── */
  function step(dt) {
    var sp = CONFIG.speed * (0.5 + 0.5 * CONFIG.intensity);
    var i, o, m = S.mouse, R = CONFIG.mouseRadius;
    var decay = Math.exp(-dt * 2.5);

    for (i = 0; i < S.doodles.length; i++) {
      o = S.doodles[i];
      o.age += dt;
      o.x += o.vx * sp * dt;
      o.y += o.vy * sp * dt;
      o.rot += o.spin * sp * dt;
      if (o.age >= o.life) S.doodles[i] = newDoodle(false);
    }

    for (i = 0; i < S.nodes.length; i++) {
      o = S.nodes[i];
      if (m) {
        var dx = o.x - m.x, dy = o.y - m.y;
        var d = Math.sqrt(dx * dx + dy * dy);
        if (d < R && d > 0.01) {
          var f = (1 - d / R) * CONFIG.mouseForce * dt;
          o.px += dx / d * f;
          o.py += dy / d * f;
        }
      }
      o.px *= decay; o.py *= decay;
      o.x += (o.vx * sp + o.px) * dt;
      o.y += (o.vy * sp + o.py) * dt;
      // мягкий отскок от краёв (с запасом, чтобы узел не «залипал» на границе)
      if (o.x < -10) { o.x = -10; o.vx = Math.abs(o.vx); }
      else if (o.x > S.w + 10) { o.x = S.w + 10; o.vx = -Math.abs(o.vx); }
      if (o.y < -10) { o.y = -10; o.vy = Math.abs(o.vy); }
      else if (o.y > S.h + 10) { o.y = S.h + 10; o.vy = -Math.abs(o.vy); }
    }
  }

  function frame(t) {
    if (!S) return;
    S.raf = requestAnimationFrame(frame);
    var dt = Math.min((t - S.last) / 1000, 0.05);
    S.last = t;
    if (dt <= 0) return;
    step(dt);
    draw();
  }

  function startLoop() {
    if (!S || S.raf || S.reduced || document.hidden) return;
    S.last = performance.now();
    S.raf = requestAnimationFrame(frame);
  }
  function stopLoop() {
    if (S && S.raf) { cancelAnimationFrame(S.raf); S.raf = 0; }
  }

  /* ── события ── */
  function onVisibility() { if (document.hidden) stopLoop(); else startLoop(); }

  function onPointerMove(e) {
    if (!S.hover) return;
    var r = S.canvas.getBoundingClientRect();
    S.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
  }
  function onPointerOut(e) { if (!e.relatedTarget) S.mouse = null; }

  function onMotionChange() {
    S.reduced = mqMotion.matches;
    if (S.reduced) { stopLoop(); S.mouse = null; seedStatic(); draw(); }
    else startLoop();
  }
  function onHoverChange() { S.hover = mqHover.matches; if (!S.hover) S.mouse = null; }

  // статичный кадр: мазки на пике прозрачности
  function seedStatic() {
    S.doodles.forEach(function (s) { s.age = s.life / 2; });
  }

  function onTheme() { if (S) readColors(); }

  /* ── публичный API ── */
  function init(opts) {
    destroy();
    var canvas = document.getElementById('bg-canvas');
    if (!canvas || !canvas.getContext) return;
    if (opts) {
      ['intensity', 'speed'].forEach(function (key) {
        if (typeof opts[key] === 'number') CONFIG[key] = opts[key];
      });
      if (opts.nodes) CONFIG.nodes = opts.nodes;
      if (opts.doodles) CONFIG.doodles = opts.doodles;
    }
    S = {
      canvas: canvas, ctx: canvas.getContext('2d'),
      w: 0, h: 0, dpr: 1, nodes: [], doodles: [], colors: [], accent: '', ink: '', dark: false,
      baseNodes: rndInt(CONFIG.nodes[0], CONFIG.nodes[1]),
      baseDoodles: rndInt(CONFIG.doodles[0], CONFIG.doodles[1]),
      mouse: null, raf: 0, last: 0, resizeTimer: 0,
      reduced: mqMotion.matches, hover: mqHover.matches
    };
    canvas.style.display = '';
    readColors();
    resize();
    if (S.reduced) { seedStatic(); draw(); } else startLoop();

    S.mo = new MutationObserver(onTheme);
    S.mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    mqDark.addEventListener('change', onTheme);
    mqMotion.addEventListener('change', onMotionChange);
    mqHover.addEventListener('change', onHoverChange);
    window.addEventListener('resize', onResize);
    if (window.ResizeObserver) { S.ro = new ResizeObserver(onResize); S.ro.observe(canvas); }
    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    window.addEventListener('pointerout', onPointerOut);
  }

  function destroy() {
    if (!S) return;
    stopLoop();
    clearTimeout(S.resizeTimer);
    S.mo.disconnect();
    if (S.ro) S.ro.disconnect();
    mqDark.removeEventListener('change', onTheme);
    mqMotion.removeEventListener('change', onMotionChange);
    mqHover.removeEventListener('change', onHoverChange);
    window.removeEventListener('resize', onResize);
    document.removeEventListener('visibilitychange', onVisibility);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerout', onPointerOut);
    S.ctx.clearRect(0, 0, S.canvas.width, S.canvas.height);
    S = null;
  }

  window.LuminaBg = { init: init, destroy: destroy, config: CONFIG };

  if (AUTO) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { init(); });
    else init();
  }
})();
