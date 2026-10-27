// 카메라·프레임 전송 (담당: B)
// 합의: JPEG 바이너리 그대로, 긴 변 1280px, 품질 0.8, 초당 2~3장
//
// - 'session:ready'(mobile-session.js)를 받은 뒤 mobileSession.wsUrl로 접속
// - 휴대폰 방향에 맞춰 프레임을 세워서 보냄 (자동 회전이 꺼져 있으면 가로로 들어도 영상이 세로로 들어옴)
// - 송신 버퍼가 비지 않았으면 그 프레임은 건너뜀 → 네트워크가 느려도 지연이 쌓이지 않음
// - 끊기면 자동 재연결. 4401(토큰 거부)은 처음 연결이면 바로 포기하고,
//   재연결 중이면 서버가 이전 연결을 정리할 때까지(최대 REJECT_RETRY_MS) 다시 시도
// 화면 요소(회전 선택, 보내는 화면 미리보기)는 mobile.html을 고치지 않고 여기서 만들어 붙입니다.
(() => {
  const FPS = 2.5;
  const LONG_SIDE = 1280;
  const QUALITY = 0.8;
  const RECONNECT_MAX_MS = 5000;
  const REJECT_RETRY_MS = 60000;   // uvicorn 핑 타임아웃(약 40초)보다 길게
  const GRAVITY_THRESHOLD = 6;     // m/s². 이보다 기울어야 방향을 바꿈 (흔들림에 깜빡이지 않게)

  const video = document.getElementById('cam-video');
  const btn = document.getElementById('cam-start');
  const info = document.getElementById('cam-info');

  // ── 화면 요소 ────────────────────────────────────────────────────
  const controls = document.createElement('div');
  controls.style.cssText = 'display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px';
  controls.innerHTML = `
    <label>회전
      <select id="cam-rot">
        <option value="auto" selected>자동</option>
        <option value="0">0°</option><option value="90">90°</option>
        <option value="180">180°</option><option value="270">270°</option>
      </select>
    </label>
    <button id="cam-torch" type="button" hidden>플래시</button>`;
  const sentBox = document.createElement('div');
  sentBox.style.cssText = 'font-size:12px;color:var(--muted)';
  sentBox.textContent = '보내는 화면';
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'display:block;max-width:60%;max-height:30vh;margin-top:4px;border:1px solid var(--muted);border-radius:6px;background:#000';
  sentBox.appendChild(canvas);
  info.after(controls, sentBox);
  video.style.objectFit = 'contain';  // 가로로 들었을 때 미리보기가 잘리지 않게

  const rotSel = document.getElementById('cam-rot');
  const torchBtn = document.getElementById('cam-torch');
  const ctx = canvas.getContext('2d');

  // ── 상태 ─────────────────────────────────────────────────────────
  let session = null;
  let stream = null, track = null, ws = null, wakeLock = null;
  let running = false, everConnected = false, rejectedSince = 0;
  let tickTimer = null, reconnectTimer = null, backoff = 500;
  let encoding = false, torchOn = false;
  let sent = 0, skipped = 0, lastKB = 0;

  document.addEventListener('session:ready', (e) => {
    session = e.detail;
    btn.disabled = false;
  });

  // ── 회전 ─────────────────────────────────────────────────────────
  const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  let sensorRotation = 0;

  function onMotion(e) {
    const g = e.accelerationIncludingGravity;
    if (!g || g.x == null) return;
    const x = isIOS ? -g.x : g.x, y = isIOS ? -g.y : g.y;  // iOS는 축 부호가 반대
    if (Math.abs(x) > GRAVITY_THRESHOLD && Math.abs(x) > Math.abs(y)) sensorRotation = x > 0 ? 270 : 90;
    else if (Math.abs(y) > GRAVITY_THRESHOLD) sensorRotation = y > 0 ? 0 : 180;
  }

  async function startMotion() {
    // iOS 13+는 사용자 동작(버튼) 안에서 권한을 받아야 함
    if (typeof DeviceMotionEvent?.requestPermission === 'function') {
      try { if (await DeviceMotionEvent.requestPermission() !== 'granted') return; } catch { return; }
    }
    window.addEventListener('devicemotion', onMotion);
  }

  // 프레임을 시계 방향으로 몇 도 돌려 보낼지
  function currentRotation() {
    if (rotSel.value !== 'auto') return Number(rotSel.value);
    // 브라우저가 화면을 이미 돌렸으면(자동 회전 켜짐) 카메라 영상도 이미 세워져 들어옴
    const angle = screen.orientation?.angle ?? window.orientation ?? 0;
    return angle !== 0 ? 0 : sensorRotation;
  }

  // ── 카메라 ───────────────────────────────────────────────────────
  async function openCamera() {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
    track = stream.getVideoTracks()[0];
    video.srcObject = stream;
    await video.play();
    const caps = track.getCapabilities ? track.getCapabilities() : {};
    if (caps.focusMode?.includes('continuous')) {
      try { await track.applyConstraints({ advanced: [{ focusMode: 'continuous' }] }); } catch {}
    }
    torchBtn.hidden = !caps.torch;
  }

  function closeCamera() {
    stream?.getTracks().forEach((tr) => tr.stop());
    stream = track = null;
    video.srcObject = null;
    torchBtn.hidden = true;
    torchOn = false;
    torchBtn.textContent = '플래시';
  }

  torchBtn.addEventListener('click', async () => {
    if (!track) return;
    torchOn = !torchOn;
    try { await track.applyConstraints({ advanced: [{ torch: torchOn }] }); } catch { torchOn = false; }
    torchBtn.textContent = torchOn ? '플래시 끄기' : '플래시';
  });

  async function requestWakeLock() {
    try { wakeLock = await navigator.wakeLock?.request('screen'); } catch {}
  }

  // ── 소켓 ─────────────────────────────────────────────────────────
  function sendMeta() {
    if (ws?.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
      type: 'meta', video_w: video.videoWidth, video_h: video.videoHeight,
      rotation: currentRotation(), rotation_mode: rotSel.value, fps: FPS, ua: navigator.userAgent,
    }));
  }

  function connect() {
    clearTimeout(reconnectTimer);
    ws = new WebSocket(session.wsUrl);
    ws.onopen = () => {
      everConnected = true;
      rejectedSince = 0;
      backoff = 500;
      session.show('대시보드와 연결됨 · 프레임 전송 중', 'ok');
      sendMeta();
    };
    ws.onclose = (ev) => {
      if (!running) return;
      if (ev.code === 4401) {
        // 처음부터 거부 → QR 문제. 재연결 중 거부 → 서버가 아직 이전 연결을 붙잡고 있을 수 있음
        if (!everConnected) return giveUp(ev);
        rejectedSince ||= Date.now();
        if (Date.now() - rejectedSince > REJECT_RETRY_MS) return giveUp(ev);
        session.show('이전 연결 정리를 기다리는 중… 자동으로 다시 연결합니다.', 'bad');
        reconnectTimer = setTimeout(connect, 3000);
        return;
      }
      if (ev.code === 4000) {  // 같은 세션에 다른 연결이 붙어 서버가 이 연결을 끊음
        stop();
        return session.show('다른 연결이 이 세션을 이어받았습니다.', 'bad');
      }
      session.show('연결이 끊겼습니다. 자동으로 다시 연결하는 중…', 'bad');
      reconnectTimer = setTimeout(connect, backoff);
      backoff = Math.min(backoff * 2, RECONNECT_MAX_MS);
    };
  }

  function giveUp(ev) {
    stop();
    session.handleClose(ev);
    btn.disabled = ev.code === 4401;  // 토큰 거부면 QR을 다시 스캔해야 함
  }

  // ── 캡처 루프 ────────────────────────────────────────────────────
  function tick() {
    if (!running) return;
    tickTimer = setTimeout(tick, 1000 / FPS);
    if (ws?.readyState !== WebSocket.OPEN || !video.videoWidth) return;
    if (encoding || ws.bufferedAmount > 0) { skipped++; return; }
    captureAndSend();
  }

  function captureAndSend() {
    encoding = true;
    const vw = video.videoWidth, vh = video.videoHeight;
    const rot = currentRotation();
    const swap = rot === 90 || rot === 270;
    const scale = Math.min(1, LONG_SIDE / Math.max(vw, vh));
    const dw = Math.round(vw * scale), dh = Math.round(vh * scale);
    const w = swap ? dh : dw, h = swap ? dw : dh;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (rot === 90) { ctx.translate(w, 0); ctx.rotate(Math.PI / 2); }
    else if (rot === 180) { ctx.translate(w, h); ctx.rotate(Math.PI); }
    else if (rot === 270) { ctx.translate(0, h); ctx.rotate(-Math.PI / 2); }
    ctx.drawImage(video, 0, 0, dw, dh);
    canvas.toBlob((blob) => {
      encoding = false;
      if (!blob || ws?.readyState !== WebSocket.OPEN) return;
      ws.send(blob);
      sent++;
      lastKB = Math.round(blob.size / 1024);
    }, 'image/jpeg', QUALITY);
  }

  setInterval(() => {
    if (!running) return;
    info.textContent = `${canvas.width}×${canvas.height} · 회전 ${currentRotation()}°${rotSel.value === 'auto' ? '(자동)' : ''}`
      + ` · 보냄 ${sent} · 건너뜀 ${skipped} · ${lastKB} KB`;
  }, 1000);

  // 회전 설정이 바뀌면 서버 로그에도 남김
  rotSel.addEventListener('change', sendMeta);

  // ── 시작 / 정지 ──────────────────────────────────────────────────
  async function start() {
    btn.disabled = true;
    if (!navigator.mediaDevices?.getUserMedia) {
      session.showError('insecure');
      btn.disabled = false;
      return;
    }
    try {
      await openCamera();
    } catch (err) {
      session.show(err.name === 'NotAllowedError'
        ? '카메라 권한이 거부되었습니다. 브라우저 설정에서 허용해 주세요.'
        : '카메라를 열 수 없습니다: ' + err.message, 'bad');
      btn.disabled = false;
      return;
    }
    startMotion();
    requestWakeLock();
    running = true;
    everConnected = false;
    rejectedSince = 0;
    btn.textContent = '정지';
    btn.disabled = false;
    connect();
    tick();
  }

  function stop() {
    running = false;
    clearTimeout(tickTimer);
    clearTimeout(reconnectTimer);
    if (ws) { ws.onclose = null; ws.close(); ws = null; }
    closeCamera();
    wakeLock?.release().catch(() => {});
    wakeLock = null;
    encoding = false;
    btn.textContent = '카메라 시작';
  }

  btn.addEventListener('click', () => {
    if (running) {
      stop();
      session.show('정지했습니다. "카메라 시작"을 누르면 다시 보냅니다.', 'ok');
    } else {
      start();
    }
  });

  // 탭이 다시 보이면 화면 꺼짐 방지를 다시 요청 (백그라운드로 가면 풀림)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && running) requestWakeLock();
  });
})();
