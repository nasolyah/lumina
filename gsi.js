// ── ВХОД ЧЕРЕЗ GOOGLE IDENTITY SERVICES ──────────────────────────────────────
// Окно Google открывается на нашем домене (luminalm.xyz), а не через редирект на
// *.supabase.co — поэтому Google показывает наш сайт, а после проверки бренда —
// название «Lumina». Google отдаёт ID-токен → Supabase signInWithIdToken (nonce:
// Google получает SHA-256 от случайной строки, Supabase — саму строку).
//
// Нет CLIENT_ID, не загрузился скрипт Google (блокировщик), нет crypto.subtle →
// init() вернёт false, и страница оставит прежнюю кнопку (OAuth-редирект Supabase).
//
// CLIENT_ID — публичный идентификатор веб-клиента OAuth из Google Cloud (тот же,
// что указан в Supabase → Authentication → Providers → Google). Не секрет.
window.LuminaGSI = (function () {
  const CLIENT_ID = '';

  let sb = null, onError = null, rawNonce = null, ready = false;

  const hex = bytes => [...bytes].map(x => x.toString(16).padStart(2, '0')).join('');
  async function sha256hex(s) {
    return hex(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s))));
  }
  function loadScript() {
    return new Promise((resolve, reject) => {
      if (window.google && google.accounts && google.accounts.id) return resolve();
      const s = document.createElement('script');
      s.src = 'https://accounts.google.com/gsi/client';
      s.async = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error('gsi: script failed to load'));
      document.head.appendChild(s);
    });
  }

  async function onCredential(resp) {
    const { error } = await sb.auth.signInWithIdToken({ provider: 'google', token: resp.credential, nonce: rawNonce });
    if (error && onError) onError(error);
    // успех → страница сама реагирует на onAuthStateChange('SIGNED_IN')
  }

  // opts: { supabase, onError(err) } → true, если кнопку Google можно рисовать
  async function init(opts) {
    sb = opts.supabase; onError = opts.onError || null;
    if (!CLIENT_ID || !sb || !(window.crypto && crypto.subtle)) return false;
    try { await loadScript(); } catch (_) { return false; }
    rawNonce = hex(crypto.getRandomValues(new Uint8Array(16)));
    google.accounts.id.initialize({
      client_id: CLIENT_ID,
      nonce: await sha256hex(rawNonce),
      callback: onCredential,
      ux_mode: 'popup',
      itp_support: true,
      use_fedcm_for_button: true,
    });
    ready = true;
    return true;
  }

  // Официальная кнопка Google (своя кнопка не может получить ID-токен).
  function render(el, { theme = 'light', lang = 'ru', width = 320 } = {}) {
    if (!ready || !el) return false;
    el.innerHTML = '';
    google.accounts.id.renderButton(el, {
      type: 'standard', size: 'large', text: 'continue_with', shape: 'rectangular',
      logo_alignment: 'center', theme: theme === 'dark' ? 'filled_black' : 'outline',
      locale: lang, width: Math.max(200, Math.min(400, Math.round(width))),
    });
    return true;
  }

  return { init, render, get ready() { return ready; } };
})();
