import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import json

# FIX: use "import keras" directly (tensorflow.keras fails on some Windows installs)
import keras

# Gemini (optional)
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "final_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "session_scaler.pkl")

WINDOW_SIZE = 4

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

if HAS_GENAI and GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None

# ---------- MODEL ----------
model = keras.models.load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ---------- STORAGE ----------
session_windows = {}
session_text_history = {}
session_question = {}

# ---------- FEATURES ----------
def compute_features(events):
    if not events:
        return np.zeros(7, dtype=np.float32), {"pause_count": 0, "backspace_count": 0}

    pending_down = {}
    dwells = []
    flights = []
    down_times = []

    backspace_count = 0
    pause_count = 0

    for e in events:
        key = e["key"]
        t = e["time"]

        if e["type"] == "down":
            pending_down[key] = t
            down_times.append(t)
            if key == "Backspace":
                backspace_count += 1

        elif e["type"] == "up":
            if key in pending_down:
                dwell = t - pending_down.pop(key)
                if dwell > 0:
                    dwells.append(dwell)

    for i in range(1, len(down_times)):
        gap = down_times[i] - down_times[i - 1]
        if gap > 0:
            flights.append(gap)
        if gap > 500:
            pause_count += 1

    dwells_s = [d / 1000.0 for d in dwells] if dwells else [0.0]
    flights_s = [f / 1000.0 for f in flights] if flights else [0.0]

    avg_dwell = np.mean(dwells_s)
    avg_flight = np.mean(flights_s)
    std_dwell = np.std(dwells_s)
    std_flight = np.std(flights_s)

    if down_times and len(down_times) > 1:
        duration_s = (down_times[-1] - down_times[0]) / 1000.0
        typing_speed = len(down_times) / max(duration_s, 0.01)
    else:
        typing_speed = len(down_times)

    features = np.array([
        avg_dwell, avg_flight, typing_speed,
        pause_count, backspace_count,
        std_dwell, std_flight
    ], dtype=np.float32)

    return features, {"pause_count": pause_count, "backspace_count": backspace_count}

# ---------- PRED ----------
def predict_load(seq):
    X = scaler.transform(seq.reshape(-1, 7)).reshape(1, 4, 7)
    p = model.predict(X, verbose=0)[0]
    return int(np.argmax(p)), p

# ---------- PROBLEM ----------
def find_problem(hist):
    if len(hist) < 2:
        return ""
    p, c = hist[-2], hist[-1]
    i = 0
    while i < min(len(p), len(c)) and p[i] == c[i]:
        i += 1
    return c[max(0, i - 40):i + 40]

# ---------- GEMINI ----------
def gen_hint(text, problem, question):
    if gemini_client is None:
        return {"hint": "Try explaining your idea more clearly.", "reference": "Start with the main point."}

    prompt = f"""
You are an intelligent writing assistant.

Question:
{question}

User text:
{text}

Problem:
{problem}

Instructions:
- Ignore grammar mistakes
- Focus on idea clarity
- Help user continue writing

Return JSON:
{{
 "hint":"short actionable hint",
 "reference":"example continuation sentence"
}}
"""

    try:
        r = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        ).text.strip()
        if r.startswith("```"):
            r = r.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(r)
    except Exception as e:
        print(f"Gemini error: {e}")
        return {"hint": "Clarify your idea", "reference": "Explain step by step."}

# ---------- SERVE index.html ----------
@app.route("/")
def serve_index():
    return send_from_directory(BASE_DIR, "index.html")

# ---------- ROUTE ----------
@app.route("/keystrokes", methods=["POST"])
def keystrokes():
    d = request.get_json()

    sid = d["session_id"]
    text = d["text"]
    events = d["events"]
    question = d.get("question", "")
    is_idle = d.get("is_idle", False)

    if sid not in session_windows:
        session_windows[sid] = []
        session_text_history[sid] = []
        session_question[sid] = question

    session_text_history[sid].append(text)

    # IDLE
    if is_idle:
        problem = "User paused for more than 10 seconds"
        res = gen_hint(text, problem, question)

        return jsonify({
            "action": "hint",
            "cognitive_load": 0,
            "problem_area": problem,
            "hint": res["hint"],
            "reference": res["reference"]
        })

    # Normal flow
    features, meta = compute_features(events)
    session_windows[sid].append(features)

    if len(session_windows[sid]) > WINDOW_SIZE:
        session_windows[sid].pop(0)

    if len(session_windows[sid]) < WINDOW_SIZE:
        return jsonify({"status": "collecting"})

    seq = np.array(session_windows[sid])
    label, prob = predict_load(seq)

    action = "hint" if label == 2 else "silent"

    hint = reference = problem = ""

    if action == "hint":
        problem = find_problem(session_text_history[sid])
        res = gen_hint(text, problem, session_question[sid])
        hint = res["hint"]
        reference = res["reference"]

    return jsonify({
        "action": action,
        "cognitive_load": label,
        "problem_area": problem,
        "hint": hint,
        "reference": reference
    })

if __name__ == "__main__":
    print(f"\n  Model loaded from: {MODEL_PATH}")
    print(f"  Scaler loaded from: {SCALER_PATH}")
    print(f"  Gemini: {'connected' if gemini_client else 'not configured (hints will be generic)'}")
    print(f"\n  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000)
