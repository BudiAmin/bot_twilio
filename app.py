import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import json
import random
import os
from flask import Flask, request, render_template, send_from_directory, abort
from keras.models import load_model
from twilio.twiml.messaging_response import MessagingResponse

# Download NLTK data (run once)
nltk.download("popular")
lemmatizer = WordNetLemmatizer()

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
FILES_SUBFOLDER = "styles/files"
FULL_FILES_PATH = os.path.join(STATIC_FOLDER, FILES_SUBFOLDER)
YOUR_PUBLIC_BOT_URL = " https://7bc486a43ba8.ngrok-free.app"

# --- Load Model dan Data ---
try:
    model = load_model(os.path.join(MODEL_DIR, "model.h5"))
    with open(os.path.join(MODEL_DIR, "intents_bot_citra.json"), encoding="utf-8") as f:
        intents = json.load(f)
    words = pickle.load(open(os.path.join(MODEL_DIR, "texts.pkl"), "rb"))
    classes = pickle.load(open(os.path.join(MODEL_DIR, "label.pkl"), "rb"))
    print("✅ Model dan data berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model atau data: {e}")
    exit()

# --- Inisialisasi Flask ---
app = Flask(__name__, static_folder=STATIC_FOLDER)

# --- Preprocessing ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in sentence_words]

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# --- Support multi response ---
def getResponse(ints, intents_json):
    if not ints:
        return [{"type": "text", "content": "Maaf, saya tidak mengerti. Coba ulangi dengan kata lain."}]
    
    tag = ints[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            responses = intent.get("responses", [])
            result = []

            for res in responses:
                if isinstance(res, str):
                    if res.startswith("FILE:"):
                        file_path = res[len("FILE:"):].strip()
                        result.append({"type": "file", "content": file_path})
                    else:
                        result.append({"type": "text", "content": res})
                elif isinstance(res, dict):
                    if "type" in res and "content" in res:
                        result.append({"type": res["type"], "content": res["content"]})
            return result

    return [{"type": "text", "content": "Maaf, tidak ada jawaban yang cocok."}]

def chatbot_response(msg):
    intents_detected = predict_class(msg, model)
    return getResponse(intents_detected, intents)

# --- Route Web Interface ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get("msg")
    if not userText:
        return "Input kosong.", 400

    responses = chatbot_response(userText)
    output = ""
    for res in responses:
        if res["type"] == "text":
            output += f"<p>{res['content']}</p>"
        elif res["type"] == "file":
            file_url = f"/{res['content']}"
            file_name = os.path.basename(res["content"])
            output += f"<a href='{file_url}' target='_blank'>📎 Unduh: {file_name}</a><br>"
    return output

# --- File Route ---
@app.route(f"/{FILES_SUBFOLDER}/<path:filename>")
def serve_static_files(filename):
    if ".." in filename:
        abort(403)
    return send_from_directory(directory=FULL_FILES_PATH, path=filename, as_attachment=False)

# --- WhatsApp Webhook ---
@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "").strip()
    sender = request.values.get("From", "")
    print(f"📩 Pesan masuk dari {sender}: {incoming_msg}")

    response = MessagingResponse()
    responses = chatbot_response(incoming_msg)

    for res in responses:
        if res["type"] == "text":
            response.message(res["content"])
        elif res["type"] == "file":
            file_url = f"{YOUR_PUBLIC_BOT_URL}/{res['content']}"
            response.message("📎 Berikut file yang Anda minta:").media(file_url)

    return str(response)

# --- Start Server ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        print(f"⚠️ Folder '{MODEL_DIR}' tidak ditemukan.")
    if not os.path.exists(FULL_FILES_PATH):
        print(f"⚠️ Folder '{FULL_FILES_PATH}' tidak ditemukan.")

    app.run(debug=True, host="0.0.0.0")
