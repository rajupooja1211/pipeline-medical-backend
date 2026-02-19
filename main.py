"""
Medical Assistant Backend — 4-Service Pipeline (Web Demo)
"""

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List
import re, os, time, base64, httpx, json

app = FastAPI(title="Medical Assistant — 4-Service Pipeline")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

conversations: Dict[str, dict] = {}
metrics_log: List[dict] = []

# ═══ SERVICE 2: Sentiment Analysis ═══
def analyze_sentiment(text: str) -> Dict[str, float]:
    t = text.lower()
    emotions = {}
    for words, key, score_per in [
        (["pain","hurt","ache","sore","throb","sharp","burning","stiff","tender"], "pain", 0.2),
        (["scared","afraid","worried","anxious","nervous","fear","terrified"], "anxiety", 0.25),
        (["sad","depressed","hopeless","crying","miserable","down","unhappy"], "sadness", 0.25),
        (["frustrated","annoyed","angry","mad","irritated"], "frustration", 0.25),
        (["tired","exhausted","fatigue","no energy","weak","drained"], "fatigue", 0.2),
        (["confused","don't know","not sure","maybe","i think"], "confusion", 0.2),
        (["better","improving","good","great","fine","okay","relieved"], "relief", 0.2),
    ]:
        score = sum(score_per for w in words if w in t)
        if score > 0:
            emotions[key] = min(score, 1.0)
    if any(w in t for w in ["really","very","so much","terrible","worst","severe"]):
        if "pain" in emotions: emotions["pain"] = min(emotions["pain"] + 0.3, 1.0)
    m = re.search(r'(\d+\.?\d*)\s*(out of|\/)\s*10', t)
    if m:
        r = float(m.group(1))
        if r >= 7:
            emotions["pain"] = max(emotions.get("pain", 0), r/10)
            emotions["distress"] = r/10
    return emotions or {"neutral": 1.0}

def get_empathy_prefix(emotions: Dict[str, float]) -> str:
    if not emotions or "neutral" in emotions: return ""
    top = max(emotions, key=emotions.get)
    s = emotions[top]
    if s < 0.2: return ""
    p = {"anxiety":"I understand this can feel worrying. ","fear":"You're in good hands. ",
         "sadness":"I'm sorry you're going through this. ","frustration":"I understand this can be frustrating. ",
         "fatigue":"I can tell you're feeling tired. ","confusion":"That's okay, take your time. ",
         "relief":"That's great to hear. "}
    if top in ["pain","distress"]:
        return "I'm really sorry about your pain. " if s >= 0.7 else "I'm sorry you're in pain. "
    return p.get(top, "")

# ═══ SERVICE 3: Medical Logic ═══
def get_next_question(session: dict, user_text: str = "") -> str:
    step = session["step"]
    if step > 0 and user_text:
        session["answers"].append({"q": step, "a": user_text})
    questions = {
        0: "Hello, I am your medical assistant. How are you feeling today? On a scale from 0 to 10, with 0 being great and 10 being in bed most of the time, how would you rate it?",
        1: "Did you encounter any morning stiffness that made it hard to move?",
        2: "How long did the stiffness linger? How long before you feel your best during the day?",
        3: "Do you feel discomfort in your upper or lower extremities?",
        4: "Can you tell me which joints are causing you pain?",
        5: "Have you noticed any new swelling in your joints?",
        6: "Any additional symptoms such as fever, fatigue, or rash?",
        7: "Would you say your symptoms are better, worse, or unchanged compared to yesterday?",
        8: "Can you grip objects like a pen or jar without difficulty?",
        9: "Did you remember to take your medications today?",
    }
    session["step"] = step + 1
    return questions.get(step, "Thank you for sharing. A doctor will review your case shortly. Take care!")

# ═══ SERVICE 1: Whisper STT ═══
async def whisper_transcribe(audio_bytes: bytes) -> tuple:
    start = time.time()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": ("audio.webm", audio_bytes, "audio/webm")},
            data={"model": "whisper-1"},
        )
    ms = (time.time() - start) * 1000
    if resp.status_code == 200:
        return resp.json().get("text", ""), ms
    print(f"❌ Whisper error: {resp.status_code} {resp.text}")
    return "", ms

# ═══ SERVICE 4: ElevenLabs TTS (non-streaming, kept for start-session) ═══
async def elevenlabs_tts(text: str, emotions: Dict[str, float] = {}) -> tuple:
    start = time.time()
    stability = 0.6 if max(emotions, key=emotions.get, default="neutral") in ["pain","distress","sadness"] else 0.5
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={"text": text, "model_id": "eleven_flash_v2_5",
                  "voice_settings": {"stability": stability, "similarity_boost": 0.75}},
        )
    ms = (time.time() - start) * 1000
    if resp.status_code != 200:
        print(f"❌ TTS error: {resp.status_code} {resp.content[:200]}")
    return (resp.content if resp.status_code == 200 else b""), ms

# ═══ ENDPOINTS ═══

# /api/web-chat — STT + Sentiment + Logic only (NO TTS). Returns JSON fast.
# Frontend calls /api/tts-stream separately for streaming audio.
@app.post("/api/web-chat")
async def web_chat(audio: UploadFile = File(...), session_id: str = Form("web-default")):
    total_start = time.time()
    if session_id not in conversations:
        conversations[session_id] = {"step": 0, "answers": []}
    session = conversations[session_id]

    audio_bytes = await audio.read()
    transcript, whisper_ms = await whisper_transcribe(audio_bytes)

    if not transcript or len(transcript.strip()) < 1:
        q = get_next_question(session, "") if session["step"] == 0 else "I didn't catch that. Could you repeat?"
        return {"transcript": "", "response": q, "emotions": {}, "metrics": {"whisper_ms": round(whisper_ms,1), "error": "no_speech"}}

    sentiment_start = time.time()
    emotions = analyze_sentiment(transcript)
    sentiment_ms = (time.time() - sentiment_start) * 1000

    logic_start = time.time()
    question = get_next_question(session, transcript)
    final_response = get_empathy_prefix(emotions) + question
    logic_ms = (time.time() - logic_start) * 1000

    total_ms = (time.time() - total_start) * 1000
    top_emotion = max(emotions, key=emotions.get)
    metrics = {
        "whisper_ms":   round(whisper_ms, 1),
        "sentiment_ms": round(sentiment_ms, 1),
        "logic_ms":     round(logic_ms, 1),
        "tts_ms":       0,   # TTS is now streamed separately
        "total_ms":     round(total_ms, 1),
        "top_emotion":  top_emotion,
        "top_score":    round(emotions.get(top_emotion, 0), 3),
    }

    print(f"\n═══ 📊 PIPELINE (no TTS) ═══")
    print(f"Whisper: {whisper_ms:.0f}ms | Sentiment: {sentiment_ms:.1f}ms | Logic: {logic_ms:.1f}ms | Total: {total_ms:.0f}ms")
    print(f"\"{transcript[:50]}\" → {top_emotion}")

    metrics_log.append({"session_id": session_id, "step": session["step"], "metrics": metrics, "transcript": transcript})

    return {"transcript": transcript, "response": final_response, "emotions": emotions, "metrics": metrics}


# /api/tts-stream — streams raw MP3 chunks from ElevenLabs directly to browser
# Browser Web Audio API plays each chunk as it arrives (~200ms to first audio)
@app.post("/api/tts-stream")
async def tts_stream(request: Request):
    body = await request.json()
    text = body.get("text", "")
    emotions = body.get("emotions", {})
    if not text:
        return {"error": "no text"}

    stability = 0.6 if emotions and max(emotions, key=emotions.get, default="neutral") in ["pain","distress","sadness"] else 0.5

    async def generate():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream",
                headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
                json={
                    "text": text,
                    "model_id": "eleven_flash_v2_5",
                    "voice_settings": {"stability": stability, "similarity_boost": 0.75},
                    "optimize_streaming_latency": 3,  # max latency optimisation
                },
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    print(f"❌ TTS stream error: {resp.status_code} {body[:200]}")
                    return
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    if chunk:
                        yield chunk

    return StreamingResponse(generate(), media_type="audio/mpeg")


# /api/start-session — greeting (still uses non-streaming TTS, one-time only)
@app.get("/api/start-session")
async def start_session(session_id: str = "web-default"):
    conversations[session_id] = {"step": 0, "answers": []}
    session = conversations[session_id]
    question = get_next_question(session, "")
    audio_resp, tts_ms = await elevenlabs_tts(question)
    audio_b64 = base64.b64encode(audio_resp).decode() if audio_resp else ""
    return {"response": question, "audio_base64": audio_b64, "metrics": {"tts_ms": round(tts_ms, 1)}}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "4-service-pipeline"}

@app.get("/metrics")
async def get_metrics():
    return {"turns": metrics_log}

@app.get("/")
async def root():
    return {"message": "4-Service Pipeline", "endpoints": ["/api/web-chat", "/api/tts-stream", "/api/start-session", "/health", "/metrics"]}
