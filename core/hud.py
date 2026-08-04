"""Fullscreen desktop HUD for JARVIS, including a live camera mode."""

from __future__ import annotations

import base64
import json
import threading

import config

_window = None
_started = False
_visible = False
_allow_close = False
_mode = "idle"
_state_lock = threading.Lock()


def _on_closing():
    if _allow_close:
        return True
    hide()
    return False


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><style>
:root {
  --cyan:#38d7ff; --cyan-soft:#8ae9ff; --blue:#087acb;
  --panel:rgba(7,18,31,.84); --line:rgba(91,213,255,.28);
}
* { box-sizing:border-box; margin:0; padding:0; }
html,body { width:100%; height:100%; overflow:hidden; }
body {
  background:
    radial-gradient(circle at 50% 46%, rgba(21,52,91,.62), transparent 48%),
    linear-gradient(rgba(52,155,211,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(52,155,211,.045) 1px, transparent 1px),
    #050810;
  background-size:100% 100%, 28px 28px, 28px 28px, 100% 100%;
  color:#d9f6ff; font-family:"Segoe UI",Arial,sans-serif; user-select:none;
}
body::after {
  content:""; position:fixed; inset:0; pointer-events:none;
  background:linear-gradient(transparent 50%,rgba(0,0,0,.07) 50%);
  background-size:100% 4px; mix-blend-mode:overlay;
}
.stage { position:absolute; inset:0; transition:opacity .55s ease, transform .55s ease; }

/* Orb shared by the idle and camera layouts */
.orb { position:relative; width:100%; height:100%; filter:drop-shadow(0 0 18px rgba(0,191,255,.48)); }
.ring { position:absolute; inset:8%; border-radius:50%; border:1px solid rgba(72,200,255,.55); }
.ring.r1 { inset:2%; border-style:dashed; animation:spin 34s linear infinite; }
.ring.r2 { inset:12%; border:3px solid transparent; border-top-color:#33d8ff; border-right-color:rgba(51,216,255,.24); animation:spin 9s linear infinite; }
.ring.r3 { inset:22%; border:5px dotted rgba(80,210,255,.5); animation:spinrev 12s linear infinite; }
.ring.r4 { inset:31%; border:2px solid #56ddff; box-shadow:inset 0 0 22px rgba(0,174,255,.35); }
.orb-core { position:absolute; inset:36%; border-radius:50%; display:grid; place-items:center;
  background:radial-gradient(circle,rgba(19,169,231,.3),rgba(4,34,66,.72));
  border:1px solid #82e8ff; letter-spacing:.34em; font-size:clamp(10px,1.15vw,19px); padding-left:.34em;
}
body[data-state="speaking"] .orb { filter:drop-shadow(0 0 28px rgba(0,220,255,.9)); }
body[data-state="speaking"] .ring.r2 { animation-duration:2.2s; }
body[data-state="speaking"] .orb-core { animation:pulse .55s ease-in-out infinite alternate; }
@keyframes spin { to { transform:rotate(360deg); } }
@keyframes spinrev { to { transform:rotate(-360deg); } }
@keyframes pulse { to { transform:scale(1.06); box-shadow:0 0 28px rgba(69,219,255,.75); } }

/* Idle layout */
#idleStage { display:flex; align-items:center; justify-content:center; }
#idleCluster { display:flex; flex-direction:column; align-items:center; transform:translateY(-2vh); }
#idleOrb { width:min(26vmin,260px); height:min(26vmin,260px); }
#clock { margin-top:24px; font-size:clamp(34px,4.2vw,68px); font-weight:600; letter-spacing:.12em; color:#f2fbff; }
#date { margin-top:8px; color:#69c7e8; letter-spacing:.35em; font-size:clamp(9px,.8vw,13px); text-transform:uppercase; }
#idleStatus { margin-top:24px; font-size:11px; letter-spacing:.45em; color:#65cceb; }
#idleStatus::before { content:"●"; color:#00ecae; margin-right:12px; }

/* Camera layout */
#cameraStage { opacity:0; pointer-events:none; transform:scale(1.02); }
body[data-mode="camera"] #idleStage { opacity:0; pointer-events:none; transform:scale(.94); }
body[data-mode="camera"] #cameraStage { opacity:1; pointer-events:auto; transform:scale(1); }
#topBar { position:absolute; left:2.3vw; right:2.3vw; top:2.1vh; height:10vh; min-height:76px;
  display:flex; align-items:center; gap:22px; padding:10px 24px;
  border:1px solid var(--line); border-radius:24px; background:linear-gradient(90deg,rgba(8,24,40,.94),rgba(8,20,34,.68));
  box-shadow:0 0 28px rgba(0,166,255,.08),inset 0 0 30px rgba(37,164,230,.035);
}
#dockOrb { width:72px; height:72px; flex:none; }
#visionTitle { letter-spacing:.3em; font-size:14px; color:#bbf0ff; }
#visionStatus { margin-top:7px; letter-spacing:.22em; font-size:10px; color:#54c9ec; }
#sessionLamp { margin-left:auto; color:#00e3ad; font-size:10px; letter-spacing:.25em; }
#sessionLamp::before { content:"●"; margin-right:9px; }
#cameraViewport { position:absolute; left:3.2vw; right:3.2vw; top:14vh; bottom:4vh;
  border:1px solid rgba(80,211,255,.46); border-radius:22px; overflow:hidden; background:#02050a;
  box-shadow:0 0 48px rgba(0,141,220,.14),inset 0 0 45px rgba(0,0,0,.72);
}
#cameraImage { width:100%; height:100%; object-fit:contain; display:block; opacity:.92; }
#cameraTint { position:absolute; inset:0; pointer-events:none;
  background:linear-gradient(120deg,rgba(0,185,255,.035),transparent 42%),
             linear-gradient(transparent 50%,rgba(30,161,224,.025) 50%);
  background-size:100% 100%,100% 5px;
}
#bbox { position:absolute; border:3px solid #00f1ff; box-shadow:0 0 13px rgba(0,232,255,.82),inset 0 0 9px rgba(0,232,255,.25); display:none; }
#bbox::before,#bbox::after { content:""; position:absolute; width:18px; height:18px; border-color:#d8fbff; }
#bbox::before { left:-5px; top:-5px; border-left:3px solid; border-top:3px solid; }
#bbox::after { right:-5px; bottom:-5px; border-right:3px solid; border-bottom:3px solid; }
#answerPanel { position:absolute; left:2.2%; right:2.2%; bottom:2.2%; min-height:90px; padding:16px 22px;
  border-left:3px solid var(--cyan); border-radius:4px 15px 15px 4px;
  background:linear-gradient(90deg,rgba(4,16,28,.94),rgba(4,15,26,.65)); backdrop-filter:blur(7px);
}
#queryText { font-size:12px; color:#74cce9; letter-spacing:.1em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
#answerText { margin-top:8px; font-size:clamp(18px,1.55vw,27px); color:#f0fcff; }
#metrics { position:absolute; right:18px; bottom:14px; font-size:10px; color:#52bedf; letter-spacing:.12em; }
#cornerTL,#cornerBR { position:absolute; width:44px; height:44px; pointer-events:none; }
#cornerTL { left:12px; top:12px; border-left:2px solid var(--cyan); border-top:2px solid var(--cyan); }
#cornerBR { right:12px; bottom:12px; border-right:2px solid var(--cyan); border-bottom:2px solid var(--cyan); }
</style></head>
<body data-mode="idle" data-state="listening">
  <section id="idleStage" class="stage">
    <div id="idleCluster">
      <div id="idleOrb" class="orb">
        <div class="ring r1"></div><div class="ring r2"></div><div class="ring r3"></div><div class="ring r4"></div>
        <div class="orb-core">JARVIS</div>
      </div>
      <div id="clock">00:00</div><div id="date">SYSTEM AT YOUR SERVICE</div>
      <div id="idleStatus">AWAITING INSTRUCTION</div>
    </div>
  </section>
  <section id="cameraStage" class="stage">
    <header id="topBar">
      <div id="dockOrb" class="orb">
        <div class="ring r1"></div><div class="ring r2"></div><div class="ring r3"></div><div class="ring r4"></div>
        <div class="orb-core">J</div>
      </div>
      <div><div id="visionTitle">JARVIS VISUAL REASONER</div><div id="visionStatus">INITIALISING CAMERA</div></div>
      <div id="sessionLamp">LIVE SENSOR</div>
    </header>
    <main id="cameraViewport">
      <img id="cameraImage" alt="JARVIS camera feed" />
      <div id="cameraTint"></div><div id="bbox"></div><div id="cornerTL"></div><div id="cornerBR"></div>
      <div id="answerPanel"><div id="queryText">Camera online. Ask JARVIS about the scene.</div>
        <div id="answerText">Visual channel ready.</div><div id="metrics"></div></div>
    </main>
  </section>
<script>
const pad=n=>String(n).padStart(2,'0');
function tick(){const d=new Date();document.getElementById('clock').textContent=`${pad(d.getHours())}:${pad(d.getMinutes())}`;
 document.getElementById('date').textContent=d.toLocaleDateString('en-GB',{weekday:'long',day:'2-digit',month:'long'}).toUpperCase();}
setInterval(tick,1000);tick();
function setMode(mode){document.body.dataset.mode=mode;}
function setState(state){document.body.dataset.state=state;document.getElementById('idleStatus').textContent=state==='speaking'?'RESPONDING':'AWAITING INSTRUCTION';}
function updateCamera(source){document.getElementById('cameraImage').src=source;}
function setVisionStatus(value){document.getElementById('visionStatus').textContent=value;}
function setTranscript(query,answer){document.getElementById('queryText').textContent=query||'';document.getElementById('answerText').textContent=answer||'';}
function setVisionResult(query,answer,bbox,confidence,latency,interaction){
 setTranscript(query,answer); const box=document.getElementById('bbox');
 if(Array.isArray(bbox)&&bbox.length===4){box.style.display='block';box.style.left=(bbox[0]*100)+'%';box.style.top=(bbox[1]*100)+'%';box.style.width=((bbox[2]-bbox[0])*100)+'%';box.style.height=((bbox[3]-bbox[1])*100)+'%';}
 else{box.style.display='none';}
 const conf=Number(confidence||0); const ms=Math.round(Number(latency||0));
 document.getElementById('metrics').textContent=`CONF ${conf.toFixed(2)} · ${ms} MS · ${interaction||''}`;
}
</script></body></html>"""


def available() -> bool:
    if not getattr(config, "HUD_ENABLED", True):
        return False
    try:
        import webview  # noqa: F401
        return True
    except Exception:
        return False


def start(main_fn) -> None:
    global _window, _started
    import webview

    _window = webview.create_window(
        "JARVIS VISION",
        html=HTML,
        width=max(960, int(getattr(config, "HUD_SIZE", 1280))),
        height=max(640, int(getattr(config, "HUD_SIZE", 720))),
        frameless=True,
        on_top=False,
        hidden=True,
        fullscreen=bool(getattr(config, "HUD_FULLSCREEN", True)),
        background_color="#050810",
    )
    _window.events.closing += _on_closing
    _started = True

    def runner():
        global _allow_close
        try:
            main_fn()
        finally:
            _allow_close = True
            try:
                _window.destroy()
            except Exception:
                pass

    webview.start(runner, debug=bool(getattr(config, "HUD_DEBUG", False)))


def _evaluate(function_name: str, *args) -> None:
    if not _started or _window is None:
        return
    payload = ",".join(json.dumps(value, ensure_ascii=False) for value in args)
    try:
        _window.evaluate_js(f"{function_name}({payload})")
    except Exception:
        pass


def show() -> None:
    global _visible
    with _state_lock:
        if not _started:
            return
        try:
            if not _visible:
                _window.show()
                _visible = True
            _evaluate("setMode", _mode)
        except Exception:
            pass


def hide() -> None:
    global _visible
    with _state_lock:
        if not _started or not _visible:
            return
        try:
            _window.hide()
            _visible = False
        except Exception:
            pass


def is_visible() -> bool:
    return _visible


def set_state(state: str) -> None:
    _evaluate("setState", state)


def enter_idle_mode() -> None:
    global _mode
    _mode = "idle"
    _evaluate("setMode", "idle")


def enter_camera_mode() -> None:
    global _mode
    _mode = "camera"
    show()
    _evaluate("setMode", "camera")


def set_vision_status(status: str) -> None:
    _evaluate("setVisionStatus", status)


def set_transcript(query: str, answer: str) -> None:
    _evaluate("setTranscript", query, answer)


def set_vision_result(
    *,
    query: str,
    answer: str,
    bbox: list[float] | None,
    confidence: float,
    latency_ms: float,
    interaction_id: str,
) -> None:
    _evaluate("setVisionResult", query, answer, bbox, confidence, latency_ms, interaction_id)


def update_camera_frame_bgr(frame_bgr, max_width: int | None = None) -> None:
    """30fps hot path: BGR in, no color conversion (cv2.imencode expects BGR).

    Budget-critical — called up to 30×/s from the camera thread. Total work here
    must stay well under 33ms: resize (INTER_AREA) + JPEG encode + base64 + one
    evaluate_js push.
    """
    if not _started or _mode != "camera":
        return
    try:
        import cv2

        image = frame_bgr
        height, width = image.shape[:2]
        limit = int(max_width or getattr(config, "VISION_DISPLAY_WIDTH", 800))
        if width > limit:
            scale = limit / width
            image = cv2.resize(image, (limit, max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(getattr(config, "VISION_DISPLAY_JPEG_QUALITY", 60))],
        )
        if not ok:
            return
        source = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
        _evaluate("updateCamera", source)
    except Exception:
        pass


def update_camera_frame(frame) -> None:
    """Back-compat wrapper: RGB in (used by preview script)."""
    try:
        import cv2

        update_camera_frame_bgr(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    except Exception:
        pass
