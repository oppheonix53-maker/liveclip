#!/usr/bin/env python3
"""liveclip: Goated AI Live & VOD Clipper (Audio Loudness + Groq Multi-Key 6k TPM + Side-by-Side GPU Render)."""
import argparse, csv, json, os, queue, re, shutil, signal, statistics, subprocess, sys, threading, time, urllib.error, urllib.request
from collections import Counter, deque

ap = argparse.ArgumentParser(description="AI-powered live stream & VOD vertical clipper.")
ap.add_argument("url", help="YouTube or Twitch link (live stream or finished VOD)")
ap.add_argument("--out", default="clips", help="output folder (default: clips)")
ap.add_argument("--quality", default="720p,720p60,480p,best", help="streamlink quality list")
ap.add_argument("--pre", type=float, default=24, help="seconds before the highlight spike")
ap.add_argument("--post", type=float, default=6, help="seconds after the highlight spike")
ap.add_argument("--spike", type=float, default=3.0, help="live chat spike multiplier")
ap.add_argument("--min-rate", type=float, default=1.5, help="minimum live chat msg/s for a spike")
ap.add_argument("--cooldown", type=float, default=60, help="minimum seconds between live clips")
ap.add_argument("--warmup", type=float, default=60, help="seconds of chat to learn baseline")
ap.add_argument("--max-clips", type=int, default=5, help="max clips to generate (0 = no limit)")
ap.add_argument("--duration", type=float, default=0, help="stop live watching after this many seconds")
ap.add_argument("--captions", choices=["hinglish", "english", "en", "hi", "auto", "off"], default="hinglish")
ap.add_argument("--model", default="auto", help="Whisper model: auto, small, medium, large-v3-turbo")
ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
ap.add_argument("--layout", choices=["blur", "crop"], default="blur")
ap.add_argument("--face", choices=["auto", "top-left", "top-right", "bottom-left", "bottom-right", "none"], default="none")
ap.add_argument("--face-pct", type=float, default=35)
ap.add_argument("--mode", choices=["auto", "live", "recorded"], default="auto")
ap.add_argument("--font", default="", help="path to a .ttf font for captions")
ap.add_argument("--groq-keys", default="", help="comma-separated Groq API keys for multi-key 6k TPM rotation")
ap.add_argument("--groq-model", default="llama-3.3-70b-versatile", help="Groq LLM model ID")
ap.add_argument("--keep", action="store_true", help="keep raw recording segments")
a = ap.parse_args()

OUT = os.path.abspath(a.out)
REC = os.path.join(OUT, ".rec")
os.makedirs(REC, exist_ok=True)
SEGLIST = os.path.join(REC, "segments.csv")
STOP = threading.Event()
QUIT = threading.Event()
LOG = os.path.join(OUT, "clips.csv")
LAST_SAY = [time.time()]


def say(*m):
    LAST_SAY[0] = time.time()
    print(time.strftime("%H:%M:%S"), *m, flush=True)


class step:
    def __init__(self, msg, every=15):
        self.msg, self.every = msg, every

    def __enter__(self):
        self.t0, self.done = time.time(), threading.Event()
        say(self.msg + "…")
        threading.Thread(target=self._tick, daemon=True).start()
        return self

    def _tick(self):
        while not self.done.wait(1):
            if time.time() - LAST_SAY[0] >= self.every:
                m, s_ = divmod(int(time.time() - self.t0), 60)
                say(f"   still {self.msg}… {m}:{s_:02d} elapsed")

    def __exit__(self, *_):
        self.done.set()


for tool in ("ffmpeg", "ffprobe"):
    if not shutil.which(tool):
        sys.exit(f"{tool} not found. Install FFmpeg first.")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
STREAMLINK = shutil.which("streamlink") or os.path.join(os.path.dirname(sys.executable), "streamlink")

# ------------------------------------------------------------------ Hardware Helpers
_NVENC = None


def has_nvenc():
    global _NVENC
    if _NVENC is None:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
        _NVENC = "h264_nvenc" in r.stdout
    return _NVENC


def get_device():
    if a.device != "auto":
        return a.device
    try:
        import ctranslate2
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


# ------------------------------------------------------------------ Groq Multi-Key 6k TPM Pool
class GroqPool:
    """Manages multiple Groq API keys with a strict 6,000 tokens/min rolling cap per key."""
    def __init__(self, keys, model="llama-3.3-70b-versatile", tpm_limit=6000):
        self.keys = [k.strip() for k in keys if k.strip()]
        self.model = model
        self.tpm_limit = tpm_limit
        self.usage = {i: deque() for i in range(len(self.keys))}
        self.lock = threading.Lock()

    def _prune(self, idx, now):
        q = self.usage[idx]
        while q and now - q[0][0] >= 61.0:
            q.popleft()
        return sum(tok for _, tok in q)

    def _acquire_key(self, est_tokens):
        while True:
            with self.lock:
                now = time.time()
                best_idx, best_used = None, 10**9
                for i in range(len(self.keys)):
                    used = self._prune(i, now)
                    if used + est_tokens <= self.tpm_limit and used < best_used:
                        best_idx, best_used = i, used
                if best_idx is not None:
                    self.usage[best_idx].append((now, est_tokens))
                    return best_idx, self.keys[best_idx]

                wait_times = []
                for i in range(len(self.keys)):
                    if self.usage[i]:
                        wait_times.append(max(1.0, 61.5 - (now - self.usage[i][0][0])))
                sleep_s = min(wait_times) if wait_times else 5.0
            say(f"⏳ Groq 6k TPM cap reached across {len(self.keys)} key(s) — waiting {sleep_s:.1f}s…")
            time.sleep(sleep_s)

    def analyze_chunk(self, chunk_text, clip_len, max_picks=3):
        """Sends one ~4.8k-token transcript chunk to Groq and returns viral clip dicts."""
        sys_prompt = (
            "You are an elite viral short-form video editor (YouTube Shorts / Reels). "
            "Given a timestamped stream transcript ([seconds] text), pick the funniest, most intense, "
            "clutch, or high-energy moments. "
            "Lines tagged [LOUD] or [VERY LOUD] indicate audio loudness spikes (screaming, shouting, hype reactions)—"
            "heavily prioritize moments where high audio energy matches funny or intense dialogue. "
            f"Each clip should be roughly {int(clip_len)} seconds long. "
            f"Return ONLY valid JSON: {{\"clips\": [{{\"start\": <int_sec>, \"end\": <int_sec>, \"score\": <1-10>, \"title\": \"<short hook>\"}}]}} "
            f"Pick up to {max_picks} best moments (score >= 7). If nothing is interesting, return {{\"clips\": []}}."
        )
        est_tokens = (len(sys_prompt) + len(chunk_text)) // 4 + 800
        payload = json.dumps({
            "model": self.model,
            "temperature": 0.3,
            "max_completion_tokens": 800,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": chunk_text},
            ],
        }).encode("utf-8")

        for _ in range(len(self.keys) * 2 + 2):
            key_idx, api_key = self._acquire_key(est_tokens)
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "liveclip/2.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    return parsed.get("clips", [])
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    say(f"⚠️ Groq Key #{key_idx + 1} hit 429 rate limit, rotating…")
                    with self.lock:
                        self.usage[key_idx].append((time.time(), self.tpm_limit))
                    time.sleep(2)
                    continue
                say(f"⚠️ Groq HTTP {e.code} on Key #{key_idx + 1}: {e.read().decode('utf-8', 'ignore')[:160]}")
                time.sleep(2)
            except Exception as e:
                say(f"⚠️ Groq error on Key #{key_idx + 1}: {e}")
                time.sleep(2)
        return []


# ------------------------------------------------------------------ 1. Live Record & Chat
rec_start = None
chat_times = deque(maxlen=200000)


def record():
    global rec_start
    shutil.rmtree(REC, ignore_errors=True)
    os.makedirs(REC, exist_ok=True)
    sl = subprocess.Popen([STREAMLINK, "--stdout", "--hls-live-edge", "2", a.url, a.quality],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
    ff = subprocess.Popen(["ffmpeg", "-v", "fatal", "-i", "pipe:0", "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
                           "-f", "segment", "-segment_time", "6", "-segment_list", SEGLIST, "-segment_list_type", "csv",
                           os.path.join(REC, "seg_%06d.ts")], stdin=sl.stdout, start_new_session=True)
    sl.stdout.close()
    while not STOP.is_set():
        if rec_start is None and os.path.exists(os.path.join(REC, "seg_000000.ts")):
            rec_start = time.time()
            say("recording started")
        if ff.poll() is not None:
            say("recording stopped")
            STOP.set()
            break
        time.sleep(0.2)
    for p in (sl, ff):
        if p.poll() is None:
            p.terminate()
    for p in (sl, ff):
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()


def segments():
    if not os.path.exists(SEGLIST):
        return []
    return [(os.path.join(REC, r[0]), float(r[1]), float(r[2])) for r in csv.reader(open(SEGLIST)) if len(r) >= 3]


def rec_now():
    return 0 if rec_start is None else time.time() - rec_start


def chat():
    from chatread import read
    while not STOP.is_set():
        try:
            for msg in read(a.url):
                if STOP.is_set():
                    return
                chat_times.append((time.time(), msg))
        except Exception as e:
            if STOP.is_set():
                return
            say(f"chat reconnecting ({e})")
            time.sleep(5)


def per_second(now, span):
    c = [0] * int(span)
    for t, _ in reversed(chat_times):
        k = int(now - t)
        if k >= span:
            break
        if k >= 0:
            c[int(span) - 1 - k] += 1
    return c


def top_words(t0, t1, n=4):
    words = Counter()
    for t, msg in chat_times:
        if t0 <= t <= t1:
            for w in set(re.findall(r"[\w']{2,}", msg.lower())):
                words[w] += 1
    return " ".join(w for w, _ in words.most_common(n))


# ------------------------------------------------------------------ 2. Whisper + Viral .ASS Captions
_whisper = None
_wlock = threading.Lock()

HINGLISH = ("Nahi yaar, yaha se rasta hi nahi hai. Bhai yeh dekho, agar aap naye ho toh follow kar lo. "
            "Main abhi helicopter dhoondh raha hoon, kya scene hai, chalo chalo. Haan bhai, mast hai.")
HINGLISH_2 = ("Haan bhai, aaj ka stream mast hone wala hai. Yeh dekho kya ho gaya, maine socha nahi tha. "
              "Chalo mission start karte hain, tum log comment mein batao kaisa laga.")
MARKERS = set("hai hain nahi nahin yaar bhai kya ka ki ke ko se mein me pe main hoon raha rahi tha toh yeh ye woh wo aur "
              "bhi hi kar karo dekho".split())


def hinglish_share(words):
    ws = [re.sub(r"[^a-z']", "", w.word.lower()) for w in words]
    ws = [w for w in ws if w]
    return sum(w in MARKERS for w in ws) / max(1, len(ws))


def whisper():
    global _whisper
    with _wlock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            dev = get_device()
            model_name = ("large-v3-turbo" if dev == "cuda" else "small") if a.model == "auto" else a.model
            say(f"loading Whisper '{model_name}' on {dev}…")
            try:
                _whisper = WhisperModel(model_name, device=dev, compute_type="float16" if dev == "cuda" else "int8")
            except Exception as e:
                if dev == "cpu":
                    raise
                say(f"GPU Whisper failed ({e}), using CPU")
                _whisper = WhisperModel("small", device="cpu", compute_type="int8")
            say(f"Whisper ready ({model_name} on {dev})")
    return _whisper


def load_wav(path):
    import wave, numpy as np
    with wave.open(path, "rb") as w:
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_words(wav_path):
    mode = a.captions
    audio = load_wav(wav_path)
    kw = dict(word_timestamps=True, vad_filter=True, condition_on_previous_text=False, beam_size=5)
    if mode in ("hinglish", "en"):
        kw.update(language="en", initial_prompt=HINGLISH)
    elif mode == "english":
        kw.update(task="translate")
    elif mode == "hi":
        kw.update(language="hi")
    segs, _ = whisper().transcribe(audio, **kw)
    words = [w for s in segs for w in (s.words or []) if w.word.strip()]
    if mode in ("hinglish", "en") and words and hinglish_share(words) < 0.08:
        segs2, _ = whisper().transcribe(audio, **{**kw, "initial_prompt": HINGLISH_2})
        words2 = [w for s in segs2 for w in (s.words or []) if w.word.strip()]
        if hinglish_share(words2) > hinglish_share(words):
            words = words2
    return words


def ass_time(s):
    h, m, sec, cs = int(s // 3600), int((s % 3600) // 60), int(s % 60), int((s - int(s)) * 100)
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"


def ts(s):
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(s % 1 * 1000):03d}"


def resolve_font():
    repo_fonts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
    if a.font and os.path.exists(a.font):
        return os.path.splitext(os.path.basename(a.font))[0].replace("-", " "), os.path.dirname(os.path.abspath(a.font))
    if os.path.isdir(repo_fonts):
        for f in os.listdir(repo_fonts):
            if f.lower().endswith(".ttf"):
                return ("Montserrat ExtraBold" if "montserrat" in f.lower() else os.path.splitext(f)[0]), repo_fonts
    return "DejaVu Sans", "/usr/share/fonts/truetype/dejavu"


def write_viral_ass(words, ass_path, srt_path, cap_mid=None, words_per_chunk=3):
    font_name, _ = resolve_font()
    WHITE, YELLOW, BLACK = "&H00FFFFFF&", "&H0000FFFF&", "&H00000000&"
    margin_v = max(120, 1920 - cap_mid - 45) if cap_mid else 420
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Viral,{font_name},84,{WHITE},{YELLOW},{BLACK},&H80000000&,-1,0,0,0,100,100,0,0,1,8,3,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= words_per_chunk or w.start - cur[-1].end > 0.55 or cur[-1].word.strip()[-1:] in ".?!"):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)

    events, srt_lines = [], []
    for idx, chunk in enumerate(chunks, 1):
        srt_lines.append(f"{idx}\n{ts(chunk[0].start)} --> {ts(chunk[-1].end)}\n{' '.join(x.word.strip() for x in chunk).strip(' ,.')}\n")
        for active_i, active_w in enumerate(chunk):
            st, en = ass_time(active_w.start), ass_time(active_w.end + 0.05)
            parts = []
            for j, w in enumerate(chunk):
                clean = w.word.strip().replace("{", "").replace("}", "").upper()
                parts.append(f"{{\\c{YELLOW}\\fscx112\\fscy112}}{clean}{{\\c{WHITE}\\fscx100\\fscy100}}" if j == active_i else clean)
            pop = "{\\t(0,65,\\fscx108\\fscy108)\\t(65,130,\\fscx100\\fscy100)}" if active_i == 0 else ""
            events.append(f"Dialogue: 0,{st},{en},Viral,,0,0,0,,{pop}{' '.join(parts)}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
    if srt_lines:
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))


# ------------------------------------------------------------------ 3. Facecam & GPU Render
FACE = None
CORNERS = {"top-left": (0, 0), "top-right": (1, 0), "bottom-left": (0, 1), "bottom-right": (1, 1)}


def probe_size(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    w, h = (int(v) for v in r.stdout.strip().split(",")[:2])
    return w, h


def face_aspect():
    return 1080 / (int(1920 * a.face_pct / 100) // 2 * 2)


def corner_face(corner, w, h):
    fw = 0.30
    fh = fw * w / face_aspect() / h
    x, y = CORNERS[corner]
    return (fw / 2 if x == 0 else 1 - fw / 2, fh / 2 if y == 0 else 1 - fh / 2, fw, fh)


def detect_face(raw, dur, w, h):
    try:
        import cv2, numpy as np
        cv2.CascadeClassifier
    except (ImportError, AttributeError):
        return None
    hc = cv2.data.haarcascades
    front = cv2.CascadeClassifier(os.path.join(hc, "haarcascade_frontalface_alt2.xml"))
    side = cv2.CascadeClassifier(os.path.join(hc, "haarcascade_profileface.xml"))
    hits, n = [], 8
    for k in range(n):
        t = dur * (k + 0.5) / n
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", raw, "-frames:v", "1", "-vf", "scale=1280:-2",
                            "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
        img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_GRAYSCALE) if r.stdout else None
        if img is None:
            continue
        img = cv2.equalizeHist(img)
        ih, iw = img.shape
        found = [tuple(f) for f in front.detectMultiScale(img, 1.08, 5, minSize=(30, 30))]
        found += [tuple(f) for f in side.detectMultiScale(img, 1.08, 5, minSize=(30, 30))]
        found += [(iw - x - w_, y, w_, h_) for (x, y, w_, h_) in side.detectMultiScale(cv2.flip(img, 1), 1.08, 5, minSize=(30, 30))]
        hits += [(k, (x + w_ / 2) / iw, (y + h_ / 2) / ih, w_ / iw) for (x, y, w_, h_) in found]
    best, best_frames = [], 0
    for _, cx, cy, _ in hits:
        group = [q for q in hits if abs(q[1] - cx) < 0.06 and abs(q[2] - cy) < 0.08]
        frames = len({q[0] for q in group})
        if (frames, len(group)) > (best_frames, len(best)):
            best, best_frames = group, frames
    if best_frames < 3:
        return None
    cx, cy = sum(q[1] for q in best) / len(best), sum(q[2] for q in best) / len(best)
    size = sorted(q[3] for q in best)[len(best) // 2]
    if 0.3 < cx < 0.7 and 0.3 < cy < 0.7 and size > 0.12:
        return None
    fw = min(0.45, max(0.18, size * 2.4))
    fh = fw * w / face_aspect() / h
    return (min(1 - fw / 2, max(fw / 2, cx)), min(1 - fh / 2, max(fh / 2, cy)), fw, fh)


def face_box(raw, dur):
    global FACE
    w, h = probe_size(raw)
    if a.face == "none":
        return None
    if a.face in CORNERS:
        f = corner_face(a.face, w, h)
    else:
        if FACE is None:
            FACE = detect_face(raw, dur, w, h) or "none"
            say("facecam: " + ("not found" if FACE == "none" else f"found at {FACE[0]:.0%}, {FACE[1]:.0%}"))
        if FACE == "none":
            return None
        f = FACE
    cx, cy, fw, fh = f
    bw, bh = int(fw * w) // 2 * 2, int(fh * h) // 2 * 2
    return (int(cx * w - bw / 2), int(cy * h - bh / 2), bw, bh)


_cut_lock = threading.Lock()


def render(raw, ss, dur, name, tmp):
    fb = face_box(raw, dur) if a.face != "none" else None
    if fb:
        x, y, bw, bh = fb
        fh = int(1920 * a.face_pct / 100) // 2 * 2
        vf = (f"[0:v]split[a][b];[a]crop={bw}:{bh}:{x}:{y},scale=1080:{fh}[f];"
              f"[b]scale=1080:{1920 - fh}:force_original_aspect_ratio=increase,crop=1080:{1920 - fh}[g];[f][g]vstack[v0]")
        cap_mid = fh
    elif a.layout == "blur":
        vf = ("[0:v]split[a][b];[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=8:2,"
              "scale=1080:1920,eq=brightness=-0.08[bg];[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]")
        cap_mid = None
    else:
        vf = "[0:v]scale=-2:1920,crop=1080:1920[v0]"
        cap_mid = None

    last = "v0"
    if a.captions != "off":
        wav = os.path.join(tmp, "a.wav")
        subprocess.run(["ffmpeg", "-v", "fatal", "-y", "-ss", f"{ss:.2f}", "-t", f"{dur:.2f}", "-i", raw,
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], check=True)
        try:
            words = transcribe_words(wav)
            if words:
                ass_path = os.path.join(tmp, "caps.ass")
                srt_path = os.path.join(OUT, name + ".srt")
                write_viral_ass(words, ass_path, srt_path, cap_mid=cap_mid)
                _, fonts_dir = resolve_font()
                ass_esc = ass_path.replace("\\", "/").replace(":", "\\:")
                fdir_esc = fonts_dir.replace("\\", "/").replace(":", "\\:")
                vf += f";[v0]ass='{ass_esc}':fontsdir='{fdir_esc}'[v1]"
                last = "v1"
        except Exception as e:
            say("captions skipped:", e)

    out = os.path.join(OUT, name + ".mp4")
    vcodec = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22", "-pix_fmt", "yuv420p"] if has_nvenc() else [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p"
    ]
    subprocess.run([
        "ffmpeg", "-v", "fatal", "-y", "-ss", f"{ss:.2f}", "-t", f"{dur:.2f}", "-i", raw,
        "-filter_complex", vf, "-map", f"[{last}]", "-map", "0:a:0?",
        *vcodec, "-r", "30", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out
    ], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


# ------------------------------------------------------------------ 4. Recorded VOD (Audio Loudness + Groq AI + Side-by-Side Render)
YTDLP = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-playlist"]
CLIENTS = [[], *[["--extractor-args", f"youtube:player_client={c}"] for c in ("tv_simply", "tv", "web_safari", "mweb", "android_vr")]]
CLIENT = []
BLOCKED = "YouTube is blocking this cloud IP right now. Try a Twitch link or retry in a few hours."


def is_bot_block(err):
    return "not a bot" in err or "Sign in to confirm" in err or "HTTP Error 429" in err


def ytdlp(args, label):
    p = subprocess.Popen([*YTDLP, *CLIENT, "--newline", "--progress-template", "download:PCT %(progress._percent_str)s", *args],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    shown, tail = -10, deque(maxlen=5)
    for line in p.stdout:
        line = line.strip()
        m = re.match(r"PCT\s*([\d.]+)%", line)
        if m:
            pct = int(float(m.group(1)))
            if pct >= shown + 10:
                shown = pct - pct % 10
                say(f"   {label}: {shown}%")
        elif line:
            tail.append(line)
    if p.wait():
        if any(is_bot_block(t) for t in tail):
            raise SystemExit(BLOCKED)
        raise RuntimeError(next((t for t in reversed(tail) if "ERROR" in t), tail[-1] if tail else "download failed"))


def is_live(url):
    global CLIENT
    for c in CLIENTS:
        r = subprocess.run([*YTDLP, *c, "--skip-download", "--print", "%(is_live)s", url], capture_output=True, text=True)
        if r.returncode == 0 or not is_bot_block(r.stderr):
            break
    else:
        raise SystemExit(BLOCKED)
    CLIENT = c
    if r.returncode:
        raise SystemExit(f"Couldn't open link: {(r.stderr.strip().splitlines() or ['error'])[-1]}")
    return r.stdout.strip().splitlines()[-1:] == ["True"]


def audio_db_curve(wav_16k):
    """Returns per-second dB above median loudness."""
    import numpy as np
    x = load_wav(wav_16k)
    sec = len(x) // 16000
    if sec < 2:
        return np.zeros(1)
    rms = np.sqrt((x[:sec * 16000].reshape(sec, 16000) ** 2).mean(1)) + 1e-6
    db = 20 * np.log10(rms)
    return db - np.median(db)


def chunk_transcript(lines, max_input_tokens=4800):
    """Packs '[sec] text' lines into ~4.8k-token chunks so total (in+out) stays under 6k TPM."""
    chunks, cur, cur_tok = [], [], 0
    for ln in lines:
        tok = len(ln) // 4 + 2
        if cur and cur_tok + tok > max_input_tokens:
            chunks.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(ln)
        cur_tok += tok
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def recorded():
    import numpy as np
    max_clips = a.max_clips or 5
    length = a.pre + a.post
    work = os.path.join(REC, "vod")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)

    keys = [k.strip() for k in a.groq_keys.split(",") if k.strip()]
    if not keys:
        sys.exit("❌ Pass at least one Groq API key via --groq-keys.")

    say(f"🚀 VOD Hybrid AI Mode: Audio Loudness + {len(keys)} Groq key(s) (6k TPM/key) + Side-by-Side GPU Render")
    audio = os.path.join(work, "audio.%(ext)s")
    with step("downloading VOD audio"):
        ytdlp(["-f", "ba/b[height<=480]/b", "-o", audio, a.url], "audio")
    got = [f for f in os.listdir(work) if f.startswith("audio.")]
    if not got:
        sys.exit("Couldn't download VOD audio.")
    wav = os.path.join(work, "all_16k.wav")
    with step("converting audio to 16kHz WAV"):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(work, got[0]),
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], check=True)

    with step("analyzing audio loudness & transcribing VOD on GPU"):
        db_curve = audio_db_curve(wav)
        segs, _ = whisper().transcribe(load_wav(wav), vad_filter=True, beam_size=1, condition_on_previous_text=False)
        lines = []
        covered_secs = set()
        for s in segs:
            txt = s.text.strip()
            if not txt:
                continue
            i0, i1 = max(0, int(s.start)), min(len(db_curve), int(s.end) + 1)
            peak_db = float(np.max(db_curve[i0:i1])) if i1 > i0 else 0.0
            tag = "[VERY LOUD] " if peak_db >= 10.0 else ("[LOUD] " if peak_db >= 6.0 else "")
            lines.append(f"[{int(s.start)}s] {tag}{txt}")
            covered_secs.update(range(i0, i1))

        # Catch pure non-verbal screams / hype bursts skipped by Whisper VAD
        loud_peaks = [i for i in range(len(db_curve)) if db_curve[i] >= 12.0 and i not in covered_secs]
        last_p = -999
        for p in loud_peaks:
            if p - last_p >= 15:
                lines.append(f"[{p}s] [VERY LOUD REACTION / SCREAM]")
                last_p = p
        lines.sort(key=lambda x: int(re.match(r"\[(\d+)s\]", x).group(1)))

    chunks = chunk_transcript(lines, max_input_tokens=4800)
    say(f"📝 Hybrid transcript ready: {len(lines)} lines -> {len(chunks)} Groq chunk(s) (~4.8k tokens each)")

    # Side-by-Side Pipeline: GPU Render Worker runs while Groq analyzes remaining chunks
    clip_q = queue.Queue()
    made_count = [0]
    seen_starts = []

    def render_worker():
        while True:
            item = clip_q.get()
            if item is None:
                clip_q.task_done()
                break
            t0, dur, score, title = item
            if made_count[0] >= max_clips:
                clip_q.task_done()
                continue
            safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_")[:28]
            name = f"clip_{int(t0 // 60)}m{int(t0 % 60):02d}s_{safe_title}"
            tmp = os.path.join(work, name)
            os.makedirs(tmp, exist_ok=True)
            raw = os.path.join(tmp, "raw.mp4")
            say(f"🎬 [Worker] Rendering '{title}' ({int(t0 // 60)}:{int(t0 % 60):02d}, score {score}/10)…")
            try:
                ytdlp(["-f", "bv*[height<=720]+ba/b[height<=720]/b", "--merge-output-format", "mp4",
                       "--download-sections", f"*{t0:.0f}-{t0 + dur:.0f}", "--force-keyframes-at-cuts",
                       "-o", raw, a.url], "clip")
                path = render(raw, 0, dur, name, tmp)
                made_count[0] += 1
                say("saved", path)
            except Exception as e:
                say(f"⚠️ [Worker] Failed '{title}': {e}")
            clip_q.task_done()

    worker = threading.Thread(target=render_worker, daemon=True)
    worker.start()

    pool = GroqPool(keys, model=a.groq_model, tpm_limit=6000)
    queued_total = 0

    for idx, ch in enumerate(chunks, 1):
        if queued_total >= max_clips or QUIT.is_set():
            break
        say(f"🤖 [Groq] Analyzing chunk {idx}/{len(chunks)}…")
        picks = pool.analyze_chunk(ch, clip_len=length, max_picks=max(1, max_clips - queued_total))
        picks.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
        for p in picks:
            if queued_total >= max_clips:
                break
            t0 = max(0.0, float(p.get("start", 0)))
            t1 = float(p.get("end", t0 + length))
            dur = max(15.0, min(60.0, t1 - t0)) if t1 > t0 else length
            if any(abs(t0 - s) < length * 0.7 for s in seen_starts):
                continue
            seen_starts.append(t0)
            queued_total += 1
            clip_q.put((t0, dur, p.get("score", 8), p.get("title", "highlight")))
            say(f"✨ [Groq] Queued #{queued_total}: '{p.get('title', 'highlight')}' at {int(t0 // 60)}:{int(t0 % 60):02d}")

    clip_q.put(None)
    worker.join()
    if not a.keep:
        shutil.rmtree(REC, ignore_errors=True)
    say(f"done: {made_count[0]} clip(s) in {OUT}")


# ------------------------------------------------------------------ 5. Live Main Loop
def main():
    say(f"liveclip -> {OUT}")
    rec_thread = threading.Thread(target=record, daemon=True)
    rec_thread.start()
    threading.Thread(target=chat, daemon=True).start()
    if a.captions != "off":
        threading.Thread(target=whisper, daemon=True).start()

    def on_ctrl_c(*_):
        if QUIT.is_set():
            STOP.set()
        QUIT.set()

    signal.signal(signal.SIGINT, on_ctrl_c)
    t_begin, last_clip, n, jobs = time.time(), -1e9, 0, []
    while not STOP.is_set() and not QUIT.is_set():
        time.sleep(1)
        if rec_start is None:
            continue
        now = time.time()
        sec = per_second(now, 160)
        warm = now - t_begin >= a.warmup and len(chat_times) >= 20
        rate = sum(sec[-8:]) / 8
        hist = sec[:-8][-int(min(152, max(8, now - t_begin - 8))):]
        bins = [sum(hist[i:i + 8]) / 8 for i in range(0, len(hist) - 7, 4)]
        base = max(statistics.median(bins) if bins else 0, 0.2)
        if warm and rate >= max(a.spike * base, a.min_rate) and rec_now() - last_clip > a.cooldown:
            last_clip = rec_now()
            n += 1
            name = time.strftime("clip_%Y%m%d_%H%M%S")
            t0, t1 = max(0, rec_now() - a.pre), rec_now() + a.post
            say(f"chat spike: clipping {t0:.0f}-{t1:.0f}s")

            def job(t0=t0, t1=t1, name=name):
                with _cut_lock:
                    segs = [s for s in segments() if s[2] > t0 and s[1] < t1]
                    if not segs:
                        return
                    tmp = os.path.join(REC, name)
                    os.makedirs(tmp, exist_ok=True)
                    lst = os.path.join(tmp, "list.txt")
                    open(lst, "w").write("".join(f"file '{f}'\n" for f, _, _ in segs))
                    raw = os.path.join(tmp, "raw.ts")
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", raw], check=True)
                    path = render(raw, max(0, t0 - segs[0][1]), min(t1, segs[-1][2]) - t0, name, tmp)
                    if path:
                        say("saved", path)

            th = threading.Thread(target=job)
            th.start()
            jobs.append(th)
        if a.max_clips and n >= a.max_clips:
            break
        if a.duration and now - t_begin > a.duration:
            break

    QUIT.set()
    for th in jobs:
        th.join()
    STOP.set()
    rec_thread.join(5)
    if not a.keep:
        shutil.rmtree(REC, ignore_errors=True)
    say(f"done: {n} clip(s) in {OUT}")


if __name__ == "__main__":
    mode = a.mode
    if mode == "auto":
        mode = "live" if is_live(a.url) else "recorded"
    recorded() if mode == "recorded" else main()
