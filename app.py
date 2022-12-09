from flask import (
    Flask, render_template, request,
    redirect, url_for, session
)
from tensorflow import keras
from random import choice
from utils import *
import numpy as np

app = Flask(__name__)
app.secret_key = "password"

@app.route("/")
def index():
    session.clear()
    return render_template("index.html")

# Training
@app.route("/learning", methods=["GET"])
def learning_get():
    message = session.get("message", "")
    letter = choice(list(ENCODER.keys()))
    return render_template("learning.html", letter=letter, message=message)

@app.route("/learning", methods=["POST"])
def learning_post():
    label = request.form["letter"]
    labels = np.load("data/labels.npy")
    labels = np.append(labels, label)
    np.save("data/labels.npy", labels)

    pixels = request.form["pixels"]
    pixels = pixels.split(",")
    img = np.array(pixels).astype(float).reshape(1, 50, 50)
    imgs = np.load("data/images.npy")
    imgs = np.vstack([imgs, img])
    np.save("data/images.npy", imgs)

    session["message"] = f"Letter {label} added to the training dataset"

    return redirect(url_for("learning_get"))

# Inference
@app.route("/practice", methods=["GET"])
def practice_get():
    message = session.get("message", "")
    return render_template("practice.html", message=message)

@app.route("/practice", methods=["POST"])
def practice_post():
    pixels = request.form["pixels"]
    pixels = pixels.split(",")
    img = np.array(pixels).astype("float").reshape(1, 50, 50, 1)

    model = keras.models.load_model("model/CNN")
    label_pred = np.argmax(model.predict(img), axis=-1)
    prediction = ENCODER.inverse[label_pred[0]]
    
    session["message"] = f"You have written a {prediction} letter!"

    return redirect(url_for("practice_get"))

if __name__ == "__main__":
    app.run(debug=True)