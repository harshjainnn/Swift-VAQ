import os
import sys
import shutil
import subprocess
from flask import Flask, request, render_template_string, redirect, url_for
import torch
import whisper
from transformers import pipeline
from pytube import YouTube

# ---------- Config ----------
DOWNLOAD_DIR = "downloads"
AUDIO_FILE = "audio.wav"
VIDEO_FILE = "video.mp4"
WHISPER_MODEL = "base"  # change to "base" or "tiny" for speed or "medium" for accuracy

# ---------- Flask app ----------
app = Flask(__name__)

# ---------- Utility functions ----------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def cleanup_temp_files(*files):
    for f in files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

# ---------- Downloaders ----------
def download_with_ytdlp(url, output_path="downloads", filename="video.mp4"):
    ensure_dir(output_path)
    file_path = os.path.join(output_path, filename)
    command = [sys.executable, "-m", "yt_dlp", "-f", "best[ext=mp4]", "-o", file_path, url]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not os.path.exists(file_path):
        # expose stderr for debugging
        raise RuntimeError(result.stderr.decode(errors="ignore") or "yt-dlp failed without stderr")
    return file_path

def download_with_pytube(url, output_path="downloads", filename="video.mp4"):
    ensure_dir(output_path)
    file_path = os.path.join(output_path, filename)
    yt = YouTube(url)
    # prefer progressive mp4 stream (audio+video) for simplicity
    stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
    if not stream:
        raise RuntimeError("No mp4 progressive stream found via pytube.")
    stream.download(output_path=output_path, filename=filename)
    if not os.path.exists(file_path):
        raise RuntimeError("pytube download failed.")
    return file_path

def download_youtube_video(url, output_path=DOWNLOAD_DIR, filename=VIDEO_FILE):
    # Try yt-dlp first (via python -m to avoid PATH problems)
    try:
        return download_with_ytdlp(url, output_path=output_path, filename=filename)
    except Exception as e:
        # fallback to pytube
        print("yt-dlp failed, falling back to pytube:", e)
        return download_with_pytube(url, output_path=output_path, filename=filename)

# ---------- Audio extraction ----------
def extract_audio(video_path, audio_path=AUDIO_FILE):
    abs_video = os.path.abspath(video_path)
    abs_audio = os.path.abspath(audio_path)
    command = [
        "ffmpeg",
        "-y",
        "-i", abs_video,
        "-ar", "16000",
        "-ac", "1",
        abs_audio
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not os.path.exists(abs_audio):
        raise RuntimeError("ffmpeg failed:\n" + result.stderr.decode(errors="ignore"))
    return abs_audio


# ---------- Whisper transcription ----------
def transcribe_audio(audio_path, model_size=WHISPER_MODEL):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(audio_path, fp16=(device=="cuda"))
    return result["text"]

# ---------- chunk helper ----------
def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ---------- Summarization & QA ----------
# instantiate pipelines lazily to avoid heavy import at server start if not used
def get_summarizer():
    # light/fast summarizer
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def get_qa_model():
    return pipeline("question-answering", model="deepset/tinyroberta-squad2")

def summarize_transcript(transcript):
    summarizer = get_summarizer()
    chunks = chunk_text(transcript, chunk_size=1000)
    summaries = summarizer(chunks, max_length=120, min_length=40, do_sample=False)
    combined = " ".join([s["summary_text"] for s in summaries])
    return combined

def answer_question(transcript, question):
    qa = get_qa_model()
    chunks = chunk_text(transcript, chunk_size=800)
    best = {"score": 0.0, "answer": "No good answer found."}
    for c in chunks:
        try:
            res = qa(question=question, context=c)
            if res.get("score", 0) > best["score"]:
                best = res
        except Exception:
            continue
    return best.get("answer", "No answer found.")

def process_question(transcript, question):
    keywords = ["about", "summary", "summarize", "overview", "main idea", "topic"]
    if any(k in question.lower() for k in keywords):
        return summarize_transcript(transcript)
    return answer_question(transcript, question)

# ---------- Web routes ----------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Video Q&A</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 30px auto; padding: 0 20px; }
    textarea, input { width: 100%; box-sizing: border-box; padding: 8px; margin: 6px 0; }
    button { padding: 10px 18px; font-size: 16px; }
    .result { background:#f3f3f3; padding:12px; border-radius:6px; margin-top:12px; white-space:pre-wrap; }
    label { font-weight:600; display:block; margin-top:10px; }
  </style>
</head>
<body>
  <h1>Video Q&amp;A</h1>
  <form method="post" action="/process">
    <label>YouTube URL</label>
    <input name="url" placeholder="https://www.youtube.com/watch?v=..." required />
    <label>Question (or type 'Summarize this video')</label>
    <input name="question" placeholder="What is this video about?" required />
    <div style="margin-top:12px;">
      <button type="submit">Process Video</button>
    </div>
  </form>
  {% if error %}
    <div class="result" style="color: #8b0000;">Error: {{ error }}</div>
  {% endif %}
  {% if transcript %}
    <h3>Transcript (preview)</h3>
    <div class="result">{{ transcript }}</div>
  {% endif %}
  {% if answer %}
    <h3>Answer / Summary</h3>
    <div class="result">{{ answer }}</div>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/process", methods=["POST"])
def process():
    url = request.form.get("url", "").strip()
    question = request.form.get("question", "").strip()
    if not url or not question:
        return render_template_string(INDEX_HTML, error="Please provide both URL and question.")

    # Prepare paths
    ensure_dir(DOWNLOAD_DIR)
    video_path = os.path.join(DOWNLOAD_DIR, VIDEO_FILE)
    audio_path = AUDIO_FILE

    try:
        # download
        try:
            video_path = download_youtube_video(url, output_path=DOWNLOAD_DIR, filename=VIDEO_FILE)
        except Exception as e:
            # present user-friendly message but keep debug inside
            raise RuntimeError(f"Download failed: {e}")

        # extract audio
        try:
            audio_path = extract_audio(video_path, audio_path=AUDIO_FILE)
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {e}")

        # transcribe
        try:
            transcript = transcribe_audio(audio_path, model_size=WHISPER_MODEL)
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

        # process question
        answer = process_question(transcript, question)

        # show preview (first 2000 chars to keep page reasonable)
        preview = transcript[:2000] + ("..." if len(transcript) > 2000 else "")
        return render_template_string(INDEX_HTML, transcript=preview, answer=answer)

    except Exception as exc:
        return render_template_string(INDEX_HTML, error=str(exc))

    finally:
        # cleanup temp files
        cleanup_temp_files(os.path.join(DOWNLOAD_DIR, VIDEO_FILE), AUDIO_FILE)

# ---------- run ----------
if __name__ == "__main__":
    # if you want to run via "python app.py" directly
    app.run(host="0.0.0.0", port=5000, debug=True)

