import os
import secrets
import sqlite3
import inspect
import requests
from functools import wraps

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash

from dotenv import load_dotenv
load_dotenv()  # must run BEFORE importing chatbot/model, since chatbot.py reads
                # OPENROUTER_API_KEY at import time

# Import your modules
from chatbot import handle_message
from model import load_pipeline, predict_crop, DEFAULT_MODEL_PATH

app = Flask(__name__)

# ====================== SECRET KEY ======================
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)

DB_PATH = os.getenv("DB_PATH", "users.db")

# ====================== FEATURE SCHEMA (must match model.py) ======================
NUMERIC_FEATURES = [
    "Phosphorus", "Potassium", "Rainfall_mm",
    "Temperature_C", "Humidity_percent", "Soil_pH", "Nitrogen",
]
CATEGORICAL_FEATURES = ["Previous_Crop", "State", "Season", "Soil_Type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ====================== DATABASE ======================
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                mobile TEXT NOT NULL UNIQUE,
                state TEXT NOT NULL,
                district TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

init_db()

def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to continue.", "error")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper

# ====================== WEATHER ======================
def get_coordinates(district, state):
    """Geocode a district/state name to lat/lon using Open-Meteo's free geocoding API."""
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        r = requests.get(url, params={"name": district, "count": 5, "language": "en"}, timeout=8)
        r.raise_for_status()
        results = r.json().get("results", [])

        for res in results:
            admin1 = res.get("admin1", "")
            if admin1 and (admin1.lower() in state.lower() or state.lower() in admin1.lower()):
                return res["latitude"], res["longitude"]

        if results:
            return results[0]["latitude"], results[0]["longitude"]
    except Exception:
        pass
    return None, None


def get_weather(district=None, state=None):
    lat, lon = (None, None)
    if district and state:
        lat, lon = get_coordinates(district, state)

    if lat is None or lon is None:
        lat, lon = 22.97, 78.65  # rough centroid of India, fallback only

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,rain,wind_speed_10m",
        }
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()["current"]
        return {
            "temperature": data.get("temperature_2m", "N/A"),
            "humidity": data.get("relative_humidity_2m", "N/A"),
            "rainfall": data.get("rain", 0),
            "windspeed": data.get("wind_speed_10m", "N/A"),
        }
    except Exception:
        return {"temperature": "N/A", "humidity": "N/A", "rainfall": 0, "windspeed": "N/A"}

# ====================== MODEL ======================
PIPELINE = None
LOAD_ERROR = None
try:
    if os.path.exists(DEFAULT_MODEL_PATH):
        PIPELINE = load_pipeline(DEFAULT_MODEL_PATH)
        print("✅ Model loaded successfully")
    else:
        LOAD_ERROR = "Model file not found. Train it first."
except Exception as e:
    LOAD_ERROR = f"Model load failed: {e}"

# ====================== AUTH ROUTES ======================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        mobile = request.form.get("mobile", "").strip()
        state = request.form.get("state", "").strip()
        district = request.form.get("district", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not all([name, mobile, state, district, password]):
            flash("All fields are required.", "error")
            return redirect(url_for("register"))

        if password != confirm:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)

        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "INSERT INTO users (name, mobile, state, district, password_hash) VALUES (?, ?, ?, ?, ?)",
                    (name, mobile, state, district, password_hash)
                )
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Mobile number already registered.", "error")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        mobile = request.form.get("mobile", "").strip()
        password = request.form.get("password", "")

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            user = conn.execute("SELECT * FROM users WHERE mobile = ?", (mobile,)).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["state"] = user["state"]
            session["district"] = user["district"]
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))


# ====================== MAIN PAGES ======================
@app.route("/")
@login_required
def index():
    weather = get_weather(session.get("district"), session.get("state"))
    return render_template("index.html",
                           user_name=session.get("user_name"),
                           temperature=weather["temperature"],
                           humidity=weather["humidity"],
                           rainfall=weather["rainfall"],
                           windspeed=weather["windspeed"],
                           load_error=LOAD_ERROR)


# ====================== HEALTH ======================
@app.route("/health")
def health():
    return jsonify({"ok": True, "model_loaded": PIPELINE is not None, "error": LOAD_ERROR})


# ====================== PREDICT ======================
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if PIPELINE is None:
        return jsonify({"ok": False, "error": LOAD_ERROR or "Model not loaded"}), 503

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"ok": False, "error": "Invalid or missing JSON body"}), 400

    missing = [f for f in ALL_FEATURES if f not in payload or str(payload[f]).strip() == ""]
    if missing:
        return jsonify({"ok": False, "error": f"Missing fields: {', '.join(missing)}"}), 400

    row = {}
    try:
        for f in NUMERIC_FEATURES:
            row[f] = float(payload[f])
        for f in CATEGORICAL_FEATURES:
            row[f] = str(payload[f]).strip()
    except (TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": f"Invalid numeric value: {e}"}), 400

    try:
        prediction, top_k = predict_crop(PIPELINE, row)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Prediction failed: {e}"}), 500

    return jsonify({"ok": True, "prediction": prediction, "top_k": top_k})


# ====================== CHAT ======================
@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reply": "Please type something."})

    def predict_fn(args):
        # args: {N, P, K, temperature, humidity, ph, rainfall} from /recommend
        row = {
            "Nitrogen": args["N"],
            "Phosphorus": args["P"],
            "Potassium": args["K"],
            "Temperature_C": args["temperature"],
            "Humidity_percent": args["humidity"],
            "Soil_pH": args["ph"],
            "Rainfall_mm": args["rainfall"],
            # /recommend has no way to supply these — using session/defaults.
            "State": session.get("state", "Unknown"),
            "Previous_Crop": "Unknown",
            "Season": "Kharif",
            "Soil_Type": "Loamy",
        }
        prediction, _ = predict_crop(PIPELINE, row)
        return prediction

    try:
        if PIPELINE is None:
            reply = handle_message(message, None)
        else:
            reply = handle_message(message, predict_fn)
    except Exception as e:
        reply = f"Sorry, something went wrong: {e}"

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)