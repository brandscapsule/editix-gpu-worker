"""
Editix Ai — worker GPU RunPod
Opérations supportées :
  - "health"     : vérifie que le worker démarre (GPU + FFmpeg + modèle)
  - "normalize"  : réencode le média (orientation cuite, H.264 SDR, max 1920, 30 fps constant)
  - "derush"     : transcription Whisper + détection silences/répétitions/hésitations
                   et proposition de coupe (meilleure prise conservée)

Contrat d'entrée :
{
  "operation": "health|normalize|derush|effects",
  "inputUrl": "https://...",          // URL signée (lecture)
  "outputUrl": "https://...",         // URL signée (écriture PUT)
  "language": "fr",                   // optionnel
  "settings": { "marginMs": 100, "minSilence": 0.28 }
}

Contrat de sortie :
{
  "status": "completed",
  "outputKey": "...",                 // uniquement pour normalize
  "transcript": { "segments": [...], "words": [...] },
  "cuts": [{ "start": 0.0, "end": 4.2, "text": "..." }, ...],
  "removed": [{ "start": ..., "end": ..., "reason": "silence|repetition|hesitation" }],
  "duration": 12.4
}
"""

import difflib
import json
import os
import re
import subprocess
import tempfile
import urllib.request

import runpod

MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
HESITATIONS = re.compile(
    r"\b(euh|euhh|hum|heu|bah|ben|voilà quoi|genre|du coup|uhm|uh|um|hmm)\b", re.IGNORECASE
)

_model = None


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
    return _model


def download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "editix-worker/1"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def upload(url: str, path: str) -> None:
    with open(path, "rb") as f:
        req = urllib.request.Request(url, data=f.read(), method="PUT", headers={"Content-Type": "video/mp4", "x-upsert": "true"})
        with urllib.request.urlopen(req, timeout=600) as r:
            if r.status >= 300:
                raise RuntimeError(f"Upload result failed: {r.status}")


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def op_health(_) -> dict:
    import torch

    return {
        "status": "completed",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "whisper": MODEL_NAME,
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
    }


_NVENC: bool | None = None


def nvenc_ok() -> bool:
    """Teste réellement NVENC : listé dans ffmpeg ne veut pas dire utilisable dans le conteneur."""
    global _NVENC
    if _NVENC is None:
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        if "h264_nvenc" not in enc:
            _NVENC = False
        else:
            t = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=30:d=0.1",
                 "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, text=True,
            )
            _NVENC = t.returncode == 0
            if not _NVENC:
                print(f"[venc] NVENC indisponible, repli libx264 : {t.stderr.strip()[:300]}", flush=True)
    return _NVENC


def _venc(crf: int = 20) -> list[str]:
    if nvenc_ok():
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(crf)]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]



def op_normalize(inp: dict, tmp: str) -> dict:
    src = os.path.join(tmp, "src")
    out = os.path.join(tmp, "out.mp4")
    download(inp["inputUrl"], src)
    run_ffmpeg(
        [
            "-i", src,
            "-map", "0:v:0", "-map", "0:a?",
            "-vf", "scale='if(gt(iw,ih),min(1920,iw),-2)':'if(gt(iw,ih),-2,min(1920,ih))':force_original_aspect_ratio=decrease",
            "-r", "30", "-fps_mode", "cfr",
            *_venc(),
            "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out,
        ]
    )
    if inp.get("outputUrl"):
        upload(inp["outputUrl"], out)
    info = probe(out)
    v = next(s for s in info["streams"] if s.get("codec_type") == "video")
    return {
        "status": "completed",
        "outputKey": inp.get("outputKey"),
        "duration": float(info["format"]["duration"]),
        "width": int(v["width"]),
        "height": int(v["height"]),
    }


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", HESITATIONS.sub(" ", text.lower())).strip()


def _similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def op_derush(inp: dict, tmp: str) -> dict:
    src = os.path.join(tmp, "src")
    wav = os.path.join(tmp, "audio.wav")
    download(inp["inputUrl"], src)
    run_ffmpeg(["-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])

    settings = inp.get("settings") or {}
    margin = float(settings.get("marginMs", 100)) / 1000.0
    min_silence = float(settings.get("minSilence", 0.28))

    model = get_model()
    segments, info = model.transcribe(wav, language=inp.get("language") or None, vad_filter=True, word_timestamps=True)

    words: list[dict] = []
    segs: list[dict] = []
    for s in segments:
        entry = {"start": s.start, "end": s.end, "text": s.text.strip()}
        for w in s.words or []:
            words.append({"start": w.start, "end": w.end, "word": w.word.strip()})
        entry["hesitations"] = len(HESITATIONS.findall(s.text))
        entry["speed"] = len(entry["text"]) / max(0.001, s.end - s.start)
        segs.append(entry)

    duration = float(info.duration)

    # 1) Silence padding : on resserre chaque prise autour de la voix (marge douce)
    removed: list[dict] = []
    for i, s in enumerate(segs):
        prev_end = segs[i - 1]["end"] if i > 0 else 0.0
        s["start"] = max(prev_end, s["start"] - margin)
        next_start = segs[i + 1]["start"] if i + 1 < len(segs) else duration
        s["end"] = min(next_start, s["end"] + margin)

    # 2) Répétitions / faux départs : on garde la meilleure prise (la plus longue et fluide)
    i = 0
    while i < len(segs) - 1:
        if _similarity(segs[i]["text"], segs[i + 1]["text"]) >= 0.75:
            candidates = [segs[i], segs[i + 1]]
            j = i + 2
            while j < len(segs) and _similarity(segs[i]["text"], segs[j]["text"]) >= 0.75:
                candidates.append(segs[j])
                j += 1
            best = max(candidates, key=lambda c: (len(_norm(c["text"])), -c["hesitations"], c["speed"]))
            for c in candidates:
                if c is not best:
                    removed.append({"start": c["start"], "end": c["end"], "reason": "repetition", "text": c["text"]})
            segs[i:j] = [best]
        i += 1

    # 3) Prises puresment hésitations ou trop courtes pour tenir un sens
    segs = [s for s in segs if len(_norm(s["text"])) > 2]

    cuts = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segs]
    kept = sum(c["end"] - c["start"] for c in cuts)
    return {
        "status": "completed",
        "duration": duration,
        "language": info.language,
        "transcript": {"segments": segs, "words": words},
        "cuts": cuts,
        "removed": removed,
        "keptRatio": round(kept / duration, 3) if duration else 1.0,
    }


# ---------------------------------------------------------------------------
# Effets GPU : détourage temporel (Robust Video Matting) + bokeh, et retouche visage
# (MediaPipe Face Mesh lissé dans le temps : bouche, mâchoire, bronzage, dents).
# ---------------------------------------------------------------------------

_rvm = None
_mesh = None


def get_rvm():
    global _rvm
    if _rvm is None:
        import torch

        _rvm = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50", trust_repo=True).eval().cuda().half()
    return _rvm


def get_mesh():
    global _mesh
    if _mesh is None:
        import mediapipe as mp

        _mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return _mesh


FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
EYES = [[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]]
MOUTH_CENTER = [13, 14, 78, 308]
JAW_L, JAW_R, CHIN = 172, 397, 152


def _radial_warp(h, w, cx, cy, radius, scale, mapx, mapy):
    """Grossit (scale>1) ou réduit (scale<1) une zone circulaire, bord progressif."""
    import numpy as np

    dx, dy = mapx - cx, mapy - cy
    d = np.sqrt(dx * dx + dy * dy)
    m = d < radius
    t = np.clip(d / radius, 0, 1)
    k = 1 - (1 - 1 / scale) * (1 - t * t) ** 2
    mapx[m] = cx + dx[m] * k[m]
    mapy[m] = cy + dy[m] * k[m]


def _pinch_x(cx, cy, radius, amount, mapx, mapy, toward):
    """Déplace les pixels d'une zone horizontalement vers `toward` (affinage de mâchoire)."""
    import numpy as np

    dx, dy = mapx - cx, mapy - cy
    d = np.sqrt(dx * dx + dy * dy)
    m = d < radius
    f = (1 - np.clip(d / radius, 0, 1)) ** 2
    mapx[m] = mapx[m] - (toward - cx) * amount * f[m]


def _poly_mask(h, w, pts, feather):
    import cv2
    import numpy as np

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    if feather > 0:
        k = int(feather) * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0


def face_retouch(frame, lm, s, base_x, base_y):
    """lm : landmarks (N,2) en pixels, lissés. s : réglages -1..1."""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    face_w = float(np.linalg.norm(lm[454] - lm[234]))
    mapx, mapy = base_x.copy(), base_y.copy()
    warped = False
    mouth = s.get("mouth", 0)
    if abs(mouth) > 0.01:
        c = lm[MOUTH_CENTER].mean(0)
        _radial_warp(h, w, c[0], c[1], face_w * 0.28, 1 + 0.18 * mouth, mapx, mapy)
        warped = True
    jaw = s.get("jaw", 0)
    if abs(jaw) > 0.01:
        cx = float(lm[CHIN][0])
        for idx in (JAW_L, JAW_R, 136, 365):
            _pinch_x(lm[idx][0], lm[idx][1], face_w * 0.22, 0.12 * jaw, mapx, mapy, cx)
        warped = True
    out = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) if warped else frame

    tan = s.get("tan", 0)
    if abs(tan) > 0.01:
        skin = _poly_mask(h, w, lm[FACE_OVAL], face_w * 0.04)
        for eye in EYES:
            skin *= 1 - _poly_mask(h, w, lm[eye], face_w * 0.02)
        skin *= 1 - _poly_mask(h, w, lm[INNER_LIPS], face_w * 0.01)
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[..., 0] -= 14 * tan * skin
        lab[..., 1] += 5 * tan * skin
        lab[..., 2] += 12 * tan * skin
        out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    teeth = s.get("teeth", 0)
    if teeth > 0.01:
        inner = _poly_mask(h, w, lm[INNER_LIPS], face_w * 0.008)
        if inner.max() > 0:
            hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
            bright = np.clip((hsv[..., 2] - 90) / 80, 0, 1) * np.clip((140 - hsv[..., 1]) / 100, 0, 1)
            m = inner * bright * teeth
            lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[..., 0] += 22 * m
            lab[..., 2] -= (lab[..., 2] - 128) * 0.85 * m
            lab[..., 1] -= (lab[..., 1] - 128) * 0.5 * m
            out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out


def op_effects(inp: dict, tmp: str) -> dict:
    """Rend une version du rush avec bokeh/détourage et retouche visage, mêmes timecodes."""
    import cv2
    import numpy as np
    import torch

    s = inp.get("settings") or {}
    bokeh = float(s.get("bokeh", 0))
    cutout = s.get("background")  # None | "#rrggbb"
    face_on = any(abs(float(s.get(k, 0))) > 0.01 for k in ("mouth", "jaw", "tan", "teeth"))
    src = os.path.join(tmp, "src")
    out = os.path.join(tmp, "out.mp4")
    download(inp["inputUrl"], src)
    info = probe(src)
    v = next(x for x in info["streams"] if x.get("codec_type") == "video")
    w, h = int(v["width"]), int(v["height"])
    # ffprobe donne la taille avant rotation : on laisse ffmpeg redresser puis on relit la taille réelle.
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        rot = int(sd.get("rotation", 0) or 0)
    if abs(rot) in (90, 270):
        w, h = h, w
    w, h = w - w % 2, h - h % 2
    fps = "30"
    reader = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", src, "-vf", f"scale={w}:{h},fps=30",
                               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    venc = _venc(19)
    writer = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{w}x{h}", "-r", fps, "-i", "-", "-i", src, "-map", "0:v", "-map", "1:a?",
                               *venc, "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                               "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                              stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    model = get_rvm() if (bokeh > 0.01 or cutout) else None
    mesh = get_mesh() if face_on else None
    rec = [None] * 4
    ratio = min(1.0, 512 / max(w, h)) if max(w, h) > 512 else 1.0
    base_y, base_x = np.mgrid[0:h, 0:w].astype(np.float32)
    smooth = None
    bg_rgb = None
    if cutout:
        c = str(cutout).lstrip("#")
        bg_rgb = np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
    frame_bytes = w * h * 3
    n = 0
    while True:
        buf = reader.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        if mesh is not None:
            r = mesh.process(frame)
            if r.multi_face_landmarks:
                lm = np.array([[p.x * w, p.y * h] for p in r.multi_face_landmarks[0].landmark], np.float32)
                smooth = lm if smooth is None else smooth * 0.6 + lm * 0.4  # lissage temporel anti-tremblement
            if smooth is not None:
                frame = face_retouch(frame, smooth, s, base_x, base_y)
        if model is not None:
            with torch.no_grad():
                t = torch.from_numpy(frame).cuda().permute(2, 0, 1).unsqueeze(0).half() / 255
                _fgr, pha, *rec = model(t, *rec, downsample_ratio=ratio)
                a = pha[0, 0].float().cpu().numpy()[..., None]
            f32 = frame.astype(np.float32)
            if bg_rgb is not None:
                bg = np.broadcast_to(bg_rgb, f32.shape)
            else:
                k = int(8 + bokeh * 50) | 1
                bg = cv2.GaussianBlur(f32, (k, k), 0)
            frame = np.clip(f32 * a + bg * (1 - a), 0, 255).astype(np.uint8)
        try:
            writer.stdin.write(frame.tobytes())
        except BrokenPipeError:
            # L'encodeur s'est arrêté : on récupère son message réel au lieu d'un « Broken pipe » opaque.
            err = (writer.stderr.read().decode("utf-8", "ignore") if writer.stderr else "").strip()
            reader.kill()
            raise RuntimeError(f"Encodage interrompu par ffmpeg : {err[:400] or 'raison inconnue'}") from None
        n += 1
    writer.stdin.close()
    writer.wait()
    reader.wait()
    if writer.returncode != 0:
        err = (writer.stderr.read().decode("utf-8", "ignore") if writer.stderr else "").strip()
        raise RuntimeError(f"Encodage de la vidéo retouchée échoué : {err[:400]}")
    if inp.get("outputUrl"):
        upload(inp["outputUrl"], out)
    return {"status": "completed", "frames": n, "width": w, "height": h, "duration": n / 30}


OPS = {"health": op_health, "normalize": op_normalize, "derush": op_derush, "effects": op_effects}


def handler(event: dict) -> dict:
    job = (event.get("input") or {})
    op = job.get("operation")
    fn = OPS.get(op)
    if fn is None:
        return {"status": "failed", "error": f"Opération inconnue: {op}"}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            if op == "health":
                return op_health(job)
            return fn(job, tmp)
    except subprocess.CalledProcessError as e:
        return {"status": "failed", "error": f"FFmpeg: {e.stderr[-400:] if e.stderr else e}"}
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "error": str(e)[:400]}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
