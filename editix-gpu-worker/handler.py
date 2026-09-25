"""
Editix Ai — worker GPU RunPod
Opérations supportées :
  - "health"     : vérifie que le worker démarre (GPU + FFmpeg + modèle)
  - "normalize"  : réencode le média (orientation cuite, H.264 SDR, max 1920, 30 fps constant)
  - "derush"     : transcription Whisper + détection silences/répétitions/hésitations
                   et proposition de coupe (meilleure prise conservée)

Contrat d'entrée :
{
  "operation": "health|normalize|derush",
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
        req = urllib.request.Request(url, data=f.read(), method="PUT")
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
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
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


OPS = {"health": op_health, "normalize": op_normalize, "derush": op_derush}


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
