"""
ClipForge Backend — HuggingFace Edition
FastAPI + yt-dlp + FFmpeg + HuggingFace BLIP (free)
Deploy free on Render.com
"""

import os, json, uuid, shutil, subprocess, tempfile, base64, asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx

app = FastAPI(title="ClipForge API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WORK_DIR = Path(tempfile.gettempdir()) / "clipforge"
WORK_DIR.mkdir(exist_ok=True)
BLIP_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"


@app.get("/")
def health():
    return {"status": "ok", "service": "ClipForge API"}


@app.post("/download")
async def download_video(payload: dict):
    url = payload.get("url", "").strip()
    if not url:
        raise HTTPException(400, "No URL provided")
    job_id = str(uuid.uuid4())[:8]
    out_dir = WORK_DIR / job_id
    out_dir.mkdir()
    out_path = out_dir / "source.mp4"
    try:
        r = subprocess.run([
            "yt-dlp", "--no-playlist",
            "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", str(out_path), "--no-warnings", "--quiet", url
        ], capture_output=True, text=True, timeout=300)
        if not out_path.exists():
            mp4s = list(out_dir.glob("*.mp4"))
            if not mp4s:
                raise HTTPException(400, "Download failed: " + r.stderr[-300:])
            out_path = mp4s[0]
        return {"job_id": job_id, "duration": get_duration(str(out_path)),
                "size_mb": round(out_path.stat().st_size / 1024 / 1024, 1)}
    except subprocess.TimeoutExpired:
        raise HTTPException(408, "Download timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    out_dir = WORK_DIR / job_id
    out_dir.mkdir()
    ext = Path(file.filename).suffix or ".mp4"
    out_path = out_dir / f"source{ext}"
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"job_id": job_id, "duration": get_duration(str(out_path)),
            "size_mb": round(out_path.stat().st_size / 1024 / 1024, 1)}


@app.post("/analyze")
async def analyze(payload: dict):
    job_id   = payload.get("job_id")
    clip_n   = int(payload.get("clip_n", 10))
    fmt      = payload.get("fmt", "16:9")
    hf_token = payload.get("hf_token", "")
    if not hf_token:
        raise HTTPException(400, "HuggingFace token required")
    src = find_source(job_id)
    dur = get_duration(src)
    n   = max(10, min(24, int(dur / 7)))
    frames = extract_frames(src, dur, n)
    if not frames:
        raise HTTPException(500, "Could not extract frames")
    captioned = await caption_blip(frames, hf_token)
    clips = select_clips(captioned, dur, clip_n)
    return {"clips": clips, "duration": dur, "frames_analyzed": len(frames)}


@app.post("/cut")
async def cut_clips(payload: dict):
    job_id     = payload.get("job_id")
    clips      = payload.get("clips", [])
    fmt        = payload.get("fmt", "16:9")
    face_track = bool(payload.get("face_track", False))
    src     = find_source(job_id)
    out_dir = Path(src).parent / "clips"
    out_dir.mkdir(exist_ok=True)
    results = []
    for i, c in enumerate(clips):
        s   = float(c["start"])
        e   = float(c["end"])
        out = out_dir / f"clip_{i+1:02d}_rank{c.get('rank',i+1)}.mp4"
        if fmt == "9:16":
            vf = face_crop(src, s, e - s) if face_track else "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=720:1280"
        else:
            vf = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black"
        ok = do_cut(src, s, e - s, vf, str(out))
        if not ok:
            ok = do_cut(src, s, e - s, None, str(out))
        if ok and out.exists():
            results.append({"clip_id": i, "rank": c.get("rank", i+1),
                "file": out.name, "download_url": f"/clip/{job_id}/{out.name}",
                "start": s, "end": e, "score": c.get("score", 80),
                "reason": c.get("reason",""), "type": c.get("type","Highlight"),
                "hook": c.get("hook",""), "captions": c.get("captions",[])})
    return {"clips": results, "total": len(results)}


@app.get("/clip/{job_id}/{filename}")
def serve_clip(job_id: str, filename: str):
    path = WORK_DIR / job_id / "clips" / filename
    if not path.exists():
        raise HTTPException(404, "Clip not found")
    return FileResponse(str(path), media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.delete("/job/{job_id}")
def cleanup(job_id: str):
    d = WORK_DIR / job_id
    if d.exists():
        shutil.rmtree(d)
    return {"deleted": job_id}


# ─────────── HELPERS ───────────

def find_source(job_id: str) -> str:
    d = WORK_DIR / job_id
    if not d.exists():
        raise HTTPException(404, f"Job {job_id} not found")
    for pat in ["*.mp4","*.mkv","*.webm","*.mov","*.avi"]:
        f = list(d.glob(pat))
        if f:
            return str(f[0])
    raise HTTPException(404, "Source video not found")


def get_duration(path: str) -> float:
    r = subprocess.run(["ffprobe","-v","quiet","-print_format","json","-show_format",path],
                       capture_output=True, text=True)
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def extract_frames(src: str, duration: float, count: int) -> list:
    frames = []
    with tempfile.TemporaryDirectory() as td:
        fps = count / duration
        subprocess.run([
            "ffmpeg","-i",src,
            "-vf", f"fps={fps:.5f},scale=384:216:force_original_aspect_ratio=decrease,pad=384:216:(ow-iw)/2:(oh-ih)/2:black",
            "-vframes", str(count), "-q:v","4","-f","image2", f"{td}/f_%04d.jpg"
        ], capture_output=True, timeout=90)
        interval = duration / (count + 1)
        for i, fp in enumerate(sorted(Path(td).glob("f_*.jpg"))):
            try:
                b64 = base64.b64encode(fp.read_bytes()).decode()
                frames.append({"ts": (i+1)*interval, "b64": b64})
            except Exception:
                pass
    return frames


async def caption_blip(frames: list, token: str) -> list:
    hdrs = {"Authorization": f"Bearer {token}", "Content-Type": "image/jpeg"}
    results = []
    async with httpx.AsyncClient(timeout=40) as client:
        for frame in frames:
            img = base64.b64decode(frame["b64"])
            caption = ""
            for attempt in range(3):
                try:
                    r = await client.post(BLIP_URL, headers=hdrs, content=img)
                    if r.status_code == 503:
                        await asyncio.sleep(8)
                        continue
                    if r.status_code == 401:
                        raise HTTPException(401, "Invalid HuggingFace token")
                    if r.status_code == 200:
                        d = r.json()
                        caption = (d[0].get("generated_text","") if isinstance(d,list) else d.get("generated_text",""))
                        break
                    await asyncio.sleep(3)
                except HTTPException:
                    raise
                except Exception:
                    await asyncio.sleep(2)
            results.append({"ts": frame["ts"], "caption": caption})
    return results


HIGH_WORDS = {"running","jumping","fight","playing","dancing","cooking","laughing","crying",
    "driving","swimming","climbing","performing","singing","throwing","catching","scoring",
    "winning","celebrating","crashing","falling","rising","dramatic","intense","exciting",
    "amazing","incredible","funny","emotional","surprise","pointing","crowd","stage","sport",
    "person","man","woman","people","face","hands","group","team","close","action","movement",
    "fast","fire","water","car","ball","food","speaking","looking","camera","close-up"}

LOW_WORDS = {"empty","blank","dark","nothing","unknown","blurry","wall","floor","ceiling",
    "transition","logo","static","text"}


def score_frame(caption: str) -> int:
    cap = caption.lower()
    score = 62
    for w in HIGH_WORDS:
        if w in cap:
            score += 5
    for w in LOW_WORDS:
        if w in cap:
            score -= 8
    score += min(12, len(caption.split()))
    return max(60, min(99, score))


def make_captions(caption: str) -> list:
    words = caption.split()
    if len(words) < 3:
        return [{"t": 0, "text": caption.capitalize()}] if caption else []
    chunk = max(3, len(words) // 3)
    lines = []
    for i in range(0, min(len(words), chunk*3), chunk):
        text = " ".join(words[i:i+chunk]).capitalize()
        if text:
            lines.append({"t": i * 2, "text": text})
    return lines[:4]


def select_clips(captioned: list, duration: float, count: int) -> list:
    TYPE_MAP = [(92,"Highlight"),(83,"Action"),(75,"Key Moment"),(0,"Interesting")]
    scored = [{"ts":f["ts"],"caption":f["caption"],
               "score":score_frame(f["caption"]),
               "captions":make_captions(f["caption"])} for f in captioned]
    scored.sort(key=lambda x: x["score"], reverse=True)
    clip_dur = min(50, max(22, int(duration/count)))
    used, clips = [], []
    for f in scored:
        if len(clips) >= count:
            break
        s = max(0, int(f["ts"]) - 3)
        e = min(duration, s + clip_dur)
        if e - s < 15:
            continue
        if any(s < ue and e > us for us,ue in used):
            continue
        used.append((s, e))
        ctype = next((t for thr,t in TYPE_MAP if f["score"] >= thr), "Interesting")
        hook = " ".join(f["caption"].split()[:5]).capitalize()
        clips.append({"rank":len(clips)+1,"start":s,"end":int(e),
            "score":f["score"],"reason":f["caption"] or f"Moment at {fmt_t(f['ts'])}",
            "type":ctype,"hook":hook,"captions":f["captions"]})
    # Fallback fill
    if len(clips) < count:
        seg = duration / count
        for i in range(count):
            if len(clips) >= count:
                break
            s = int(i*seg + seg*0.1)
            e = int(min(s + clip_dur, duration))
            if e-s>15 and not any(s<ue and e>us for us,ue in used):
                used.append((s,e))
                clips.append({"rank":len(clips)+1,"start":s,"end":e,"score":70,
                    "reason":f"Segment at {fmt_t(s)}","type":"Interesting","hook":"","captions":[]})
    clips.sort(key=lambda x: x["score"], reverse=True)
    for i,c in enumerate(clips): c["rank"] = i+1
    clips.sort(key=lambda x: x["start"])
    return clips[:count]


def do_cut(src:str, start:float, dur:float, vf:str, out:str) -> bool:
    cmd = ["ffmpeg","-y","-ss",str(start),"-i",src,"-t",str(dur)]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v","libx264","-preset","fast","-crf","23",
            "-c:a","aac","-b:a","128k","-movflags","+faststart",
            "-avoid_negative_ts","make_zero", out]
    return subprocess.run(cmd, capture_output=True, timeout=120).returncode == 0


def face_crop(src:str, start:float, dur:float) -> str:
    try:
        import cv2
        with tempfile.TemporaryDirectory() as td:
            sample = f"{td}/s.jpg"
            subprocess.run(["ffmpeg","-ss",str(start+dur/2),"-i",src,
                "-vframes","1","-vf","scale=480:270","-q:v","3",sample],
                capture_output=True, timeout=10)
            img = cv2.imread(sample)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                fc = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
                faces = fc.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    fx,fy,fw,fh = max(faces, key=lambda f: f[2]*f[3])
                    xb = (fx + fw/2) / 480.0
                    return (f"crop=ih*9/16:ih:'max(0,min(iw-ih*9/16,(iw-ih*9/16)*{xb:.3f}))':0,"
                            f"scale=720:1280")
    except Exception:
        pass
    return "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=720:1280"


def fmt_t(s:float) -> str:
    s=int(s); return f"{s//60}:{s%60:02d}"
