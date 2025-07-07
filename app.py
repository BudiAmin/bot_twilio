import nltk
nltk.download("popular")

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np
import json
import random

from keras.models import load_model
from flask import Flask, render_template, request
from twilio.twiml.messaging_response import MessagingResponse

# Load Model dan Data
model = load_model("Model/model.h5")
intents = json.loads(open("Model/intents_chatbot_pompa_air.json").read())
words = pickle.load(open("Model/texts.pkl", "rb"))
classes = pickle.load(open("Model/label.pkl", "rb"))

# Inisialisasi Flask
app = Flask(__name__)
app.static_folder = "static"


# Preprocessing untuk input teks
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("ditemukan dalam bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if not ints:
        return "Maaf, saya tidak mengerti. Bisa diulang dengan kalimat berbeda?"
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


# Route untuk tampilan Web
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get("msg")
    return chatbot_response(userText)


# Route untuk WhatsApp Webhook (Twilio)
@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "")
    sender = request.values.get("From", "")
    print(f"Pesan WA masuk dari {sender}: {incoming_msg}")

    bot_reply = chatbot_response(incoming_msg)

    response = MessagingResponse()
    response.message(bot_reply)
    return str(response)


if __name__ == "__main__":
    app.run(debug=True)
