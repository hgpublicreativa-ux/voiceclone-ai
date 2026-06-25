/**
 * VoiceClone AI — Widget embebible
 * Uso: <script src="voiceclone-widget.js" data-api="https://TU-URL.railway.app"></script>
 * Genera un botón flotante que abre el panel de clonación.
 * También expone window.VoiceCloneAI.speak(text, lang) para llamar /speak directamente.
 */
(function () {
  const API = document.currentScript?.dataset?.api || 'http://localhost:8000';

  // ── Public API ──────────────────────────────────────────────────────────────
  window.VoiceCloneAI = {
    /**
     * Genera audio con la voz predeterminada guardada.
     * @param {string} text
     * @param {string} lang  'es' | 'en' | 'fr' | 'de' | 'pt' | 'it' | 'zh'
     * @returns {Promise<Blob>} WAV blob
     */
    async speak(text, lang = 'es') {
      const res = await fetch(`${API}/speak`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language: lang }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Error' }));
        throw new Error(err.detail);
      }
      return res.blob();
    },

    /**
     * Reproduce texto en la voz predeterminada.
     * @param {string} text
     * @param {string} lang
     * @returns {Promise<HTMLAudioElement>}
     */
    async playText(text, lang = 'es') {
      const blob = await this.speak(text, lang);
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.play();
      return audio;
    },

    /**
     * Abre el panel de clonación (inline, no iframe).
     */
    openPanel() {
      document.getElementById('_vc-panel')?.classList.toggle('_vc-open');
    },
  };

  // ── Styles ──────────────────────────────────────────────────────────────────
  const style = document.createElement('style');
  style.textContent = `
    #_vc-fab {
      position: fixed; bottom: 28px; right: 28px; z-index: 9999;
      width: 56px; height: 56px; border-radius: 50%;
      background: linear-gradient(135deg, #00c8ff, #7b5ea7);
      border: none; cursor: pointer; box-shadow: 0 4px 20px rgba(0,200,255,0.4);
      display: flex; align-items: center; justify-content: center;
      font-size: 22px; transition: transform 0.2s;
    }
    #_vc-fab:hover { transform: scale(1.1); }
    #_vc-panel {
      position: fixed; bottom: 96px; right: 28px; z-index: 9998;
      width: 360px; background: #070f1a; border: 1px solid rgba(0,200,255,0.25);
      border-radius: 18px; box-shadow: 0 8px 40px rgba(0,0,0,0.6);
      font-family: sans-serif; overflow: hidden;
      transform: scale(0.9) translateY(20px); opacity: 0;
      pointer-events: none; transition: all 0.25s;
    }
    #_vc-panel._vc-open { transform: scale(1) translateY(0); opacity: 1; pointer-events: all; }
    ._vc-header {
      padding: 16px 20px 12px;
      background: linear-gradient(135deg, rgba(0,200,255,0.08), rgba(123,94,167,0.08));
      border-bottom: 1px solid rgba(0,200,255,0.1);
    }
    ._vc-title { color: #00c8ff; font-size: 13px; font-weight: 700; letter-spacing: 0.08em; }
    ._vc-sub { color: rgba(180,210,230,0.5); font-size: 11px; margin-top: 2px; }
    ._vc-body { padding: 16px 20px; }
    ._vc-label { color: rgba(180,210,230,0.6); font-size: 11px; margin-bottom: 6px; letter-spacing: 0.05em; }
    ._vc-input {
      width: 100%; background: rgba(0,0,0,0.4); border: 1px solid rgba(0,200,255,0.15);
      border-radius: 10px; padding: 10px 12px; color: #e8f4f8; font-size: 13px;
      resize: vertical; min-height: 80px; box-sizing: border-box;
    }
    ._vc-input:focus { outline: none; border-color: rgba(0,200,255,0.4); }
    ._vc-row { display: flex; gap: 8px; margin-top: 10px; }
    ._vc-select {
      flex: 1; background: rgba(0,0,0,0.4); border: 1px solid rgba(0,200,255,0.15);
      border-radius: 10px; padding: 9px 10px; color: #e8f4f8; font-size: 12px;
    }
    ._vc-btn {
      flex: 2; background: linear-gradient(135deg, rgba(0,200,255,0.2), rgba(123,94,167,0.2));
      border: 1px solid rgba(0,200,255,0.3); border-radius: 10px; cursor: pointer;
      color: #e8f4f8; font-size: 12px; font-weight: 700; letter-spacing: 0.08em;
      transition: all 0.2s; padding: 9px;
    }
    ._vc-btn:hover { box-shadow: 0 0 16px rgba(0,200,255,0.25); }
    ._vc-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    ._vc-audio { width: 100%; margin-top: 12px; accent-color: #00c8ff; }
    ._vc-err { color: #ff4d6a; font-size: 11px; margin-top: 8px; display: none; }
    ._vc-err.show { display: block; }
    ._vc-status { color: rgba(0,255,163,0.8); font-size: 11px; margin-top: 8px; display: none; }
    ._vc-status.show { display: block; }
    ._vc-full-link {
      display: block; text-align: center; padding: 10px; border-top: 1px solid rgba(0,200,255,0.1);
      color: rgba(0,200,255,0.5); font-size: 11px; text-decoration: none; letter-spacing: 0.05em;
    }
    ._vc-full-link:hover { color: #00c8ff; }
  `;
  document.head.appendChild(style);

  // ── DOM ─────────────────────────────────────────────────────────────────────
  const fab = document.createElement('button');
  fab.id = '_vc-fab';
  fab.title = 'VoiceClone AI';
  fab.textContent = '🎙';
  fab.onclick = () => window.VoiceCloneAI.openPanel();

  const panel = document.createElement('div');
  panel.id = '_vc-panel';
  panel.innerHTML = `
    <div class="_vc-header">
      <div class="_vc-title">🎙 VOICECLONE AI</div>
      <div class="_vc-sub">Genera audio con voz predeterminada</div>
    </div>
    <div class="_vc-body">
      <div class="_vc-label">TEXTO A GENERAR</div>
      <textarea class="_vc-input" id="_vc-text" placeholder="Escribe el texto..."></textarea>
      <div class="_vc-row">
        <select class="_vc-select" id="_vc-lang">
          <option value="es">🇪🇸 ES</option>
          <option value="en">🇬🇧 EN</option>
          <option value="fr">🇫🇷 FR</option>
          <option value="de">🇩🇪 DE</option>
          <option value="pt">🇧🇷 PT</option>
          <option value="it">🇮🇹 IT</option>
          <option value="zh">🇨🇳 ZH</option>
        </select>
        <button class="_vc-btn" id="_vc-gen">⚡ GENERAR</button>
      </div>
      <p class="_vc-err" id="_vc-err"></p>
      <p class="_vc-status" id="_vc-status"></p>
      <audio class="_vc-audio" id="_vc-audio" controls style="display:none"></audio>
    </div>
    <a class="_vc-full-link" href="${API.replace('8000', '5500') || '#'}" target="_blank">
      Abrir panel completo →
    </a>
  `;

  document.body.appendChild(fab);
  document.body.appendChild(panel);

  // ── Logic ───────────────────────────────────────────────────────────────────
  document.getElementById('_vc-gen').onclick = async function () {
    const text = document.getElementById('_vc-text').value.trim();
    const lang = document.getElementById('_vc-lang').value;
    const err = document.getElementById('_vc-err');
    const status = document.getElementById('_vc-status');
    const audio = document.getElementById('_vc-audio');

    err.classList.remove('show');
    status.classList.remove('show');
    if (!text) { err.textContent = 'Escribe un texto.'; err.classList.add('show'); return; }

    this.disabled = true;
    this.textContent = '⏳ Generando...';
    status.textContent = 'Procesando con XTTS v2...';
    status.classList.add('show');

    try {
      const blob = await window.VoiceCloneAI.speak(text, lang);
      audio.src = URL.createObjectURL(blob);
      audio.style.display = 'block';
      audio.play();
      status.textContent = '✓ Audio listo';
    } catch (e) {
      err.textContent = e.message;
      err.classList.add('show');
      status.classList.remove('show');
    } finally {
      this.disabled = false;
      this.textContent = '⚡ GENERAR';
    }
  };
})();
