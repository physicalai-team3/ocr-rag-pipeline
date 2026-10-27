// ⑥ RAG 검색 영역 — 담당 D
// dashboard.html(A 담당)의 <section id="zone-rag"> 안만 채운다. 다른 영역은 건드리지 않는다.
// 색은 static/css/common.css 변수를 쓴다. 영역 제목(zone-head)은 A 가 만든 것을 그대로 둔다.
(() => {
  const zone = document.getElementById('zone-rag');
  if (!zone) return;

  const style = document.createElement('style');
  style.textContent = `
    #zone-rag form { display:flex; gap:8px; flex-wrap:wrap; }
    #zone-rag input { flex:1 1 220px; min-width:0; padding:10px 12px; border-radius:8px;
      border:1px solid var(--line); background:var(--bg); color:var(--ink); font-size:15px; }
    #zone-rag input:focus-visible, #zone-rag button:focus-visible { outline:2px solid var(--violet); outline-offset:1px; }
    #zone-rag .rag-go { padding:10px 14px; border:0; border-radius:8px; background:var(--violet);
      color:var(--bg); font-weight:700; cursor:pointer; }
    #zone-rag .rag-go:disabled { opacity:.5; cursor:wait; }
    #zone-rag .rag-ex { margin-top:-6px; display:flex; gap:6px; flex-wrap:wrap; }
    #zone-rag .rag-ex button { background:none; border:1px solid var(--line); color:var(--muted);
      border-radius:999px; padding:4px 9px; font-size:12px; cursor:pointer; }
    #zone-rag .rag-ex button:hover { color:inherit; }
    #zone-rag .rag-ans { padding:12px 14px; border-left:3px solid var(--violet);
      background:rgba(127,127,127,.08); border-radius:0 8px 8px 0; line-height:1.6; white-space:pre-wrap; }
    #zone-rag .rag-ans.miss { border-color:var(--line); color:var(--muted); }
    #zone-rag .rag-meta { margin-top:-10px; font:11px ui-monospace,monospace; color:var(--muted); }
    #zone-rag .rag-src { display:grid; gap:6px; }
    #zone-rag details { border:1px solid var(--line); border-radius:8px; padding:8px 10px; font-size:13px; }
    #zone-rag summary { cursor:pointer; display:flex; gap:8px; align-items:baseline; }
    #zone-rag summary b { color:var(--blue); font-family:ui-monospace,monospace; }
    #zone-rag summary span:last-child { margin-left:auto; font:11px ui-monospace,monospace; color:var(--muted); }
    #zone-rag details p { margin:6px 0 0; color:var(--muted); line-height:1.5; overflow-wrap:anywhere; }
  `;
  document.head.appendChild(style);

  zone.insertAdjacentHTML('beforeend', `
    <form id="rag-form">
      <input id="rag-q" type="text" placeholder="저장된 카드에 대해 질문하세요" autocomplete="off" maxlength="1000">
      <button class="rag-go" id="rag-go" type="submit">검색</button>
    </form>
    <div class="rag-ex" id="rag-ex"></div>
    <div class="rag-ans" id="rag-ans" hidden></div>
    <div class="rag-meta" id="rag-meta"></div>
    <div class="rag-src" id="rag-src"></div>`);

  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  // 현황 피드와 같은 출처 형식: #014 How Bread Fails
  const num = s => Number.isInteger(s.card_no) ? '#' + String(s.card_no).padStart(3, '0') : '';

  const EXAMPLES = [
    'What water temperature kills yeast?',
    'What is the packaging capacity at Plant B?',
    'When did paper reach Baghdad?',
  ];
  $('rag-ex').innerHTML = EXAMPLES.map(q => `<button type="button">${esc(q)}</button>`).join('');

  async function ask(q) {
    $('rag-go').disabled = true;
    $('rag-ans').hidden = false;
    $('rag-ans').className = 'rag-ans';
    $('rag-ans').textContent = '검색 중…';
    $('rag-meta').textContent = '';
    $('rag-src').innerHTML = '';
    try {
      const r = await fetch('/api/ask', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || r.statusText);
      $('rag-ans').className = d.found === false ? 'rag-ans miss' : 'rag-ans';
      $('rag-ans').textContent = d.answer;
      const ms = d.ms ? ` · 검색 ${d.ms.search}ms · 답변 ${d.ms.answer}ms` : '';
      $('rag-meta').textContent = `출처 ${d.sources.length}개${ms}${d.mode === 'extractive' ? ' · 원문 기반' : ''}`;
      // 답변이 인용한 카드를 펼친다 (없으면 1위)
      const cited = (d.answer.match(/\[#(\d{3})/) || [])[1];
      const openIdx = Math.max(0, d.sources.findIndex(s => num(s) === '#' + cited));
      const body = s => (s.text || '').startsWith(s.title || '\u0000') ? s.text.slice(s.title.length).trim() : (s.text || '');
      $('rag-src').innerHTML = d.sources.map((s, i) => `
        <details${i === openIdx ? ' open' : ''}>
          <summary><b>${esc(num(s))}</b><span>${esc(s.title || '제목 없음')}</span>
            <span>${typeof s.score === 'number' ? s.score.toFixed(3) : ''}</span></summary>
          ${s.text ? `<p>${esc(body(s))}</p>` : ''}
        </details>`).join('');
    } catch (e) {
      $('rag-ans').className = 'rag-ans miss';
      $('rag-ans').textContent = '오류: ' + e.message;
    } finally {
      $('rag-go').disabled = false;
    }
  }

  $('rag-form').addEventListener('submit', e => {
    e.preventDefault();
    const q = $('rag-q').value.trim();
    if (q) ask(q);
  });
  $('rag-ex').addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    $('rag-q').value = b.textContent;
    ask(b.textContent);
  });
})();
