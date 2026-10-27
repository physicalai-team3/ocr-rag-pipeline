// 휴대폰 페이지의 세션 부분 (담당: A)
//
// 주소의 sid·t를 읽어 서버에 유효한지 먼저 물어보고(토큰은 소모하지 않음),
// 괜찮으면 window.mobileSession을 만들고 'session:ready'를 보냅니다.
// camera.js(B)는 이 이벤트를 받은 뒤 mobileSession.wsUrl로 접속하고,
// 소켓이 닫히면 mobileSession.handleClose(event)를 불러 주세요.
(() => {
  const box = document.getElementById('session-msg');

  const REASONS = {
    missing: 'QR 주소가 올바르지 않습니다. 대시보드의 QR을 다시 스캔하세요.',
    invalid: '알 수 없는 세션입니다. 대시보드의 QR을 다시 스캔하세요.',
    expired: 'QR이 만료되었습니다. 대시보드에서 "새 QR"을 누르고 다시 스캔하세요.',
    rejected: '서버가 연결을 거부했습니다 (QR 만료). 대시보드의 QR을 다시 스캔하세요.',
    insecure: 'HTTPS로 접속해야 카메라를 쓸 수 있습니다. https:// 주소로 다시 여세요.',
    network: '서버에 연결할 수 없습니다. PC와 같은 Wi-Fi인지 확인하세요.',
  };

  function show(text, kind) {
    box.textContent = text;
    box.className = kind ?? '';
  }
  const showError = (reason) => show(REASONS[reason] ?? reason, 'bad');

  async function init() {
    const params = new URLSearchParams(location.search);
    const sid = params.get('sid');
    const token = params.get('t');
    if (!sid || !token) return showError('missing');
    if (!window.isSecureContext) return showError('insecure');

    let st;
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(sid)}?t=${encodeURIComponent(token)}`);
      st = await res.json();
    } catch {
      return showError('network');
    }
    if (!st.valid) return showError(st.reason);

    const q = `sid=${encodeURIComponent(sid)}&t=${encodeURIComponent(token)}`;
    window.mobileSession = {
      sid,
      token,
      wsUrl: `wss://${location.host}/ws/camera?${q}`,
      show,
      showError,
      // 4401 = 서버가 토큰을 거부. 그 외는 네트워크 끊김으로 봅니다.
      handleClose(ev) {
        if (ev.code === 4401) showError('rejected');
        else show('연결이 끊겼습니다. "카메라 시작"을 눌러 다시 연결하세요.', 'bad');
      },
    };
    show('세션 확인 완료. "카메라 시작"을 누르세요.', 'ok');
    document.dispatchEvent(new CustomEvent('session:ready', { detail: window.mobileSession }));
  }

  init();
})();
