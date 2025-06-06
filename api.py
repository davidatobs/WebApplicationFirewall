from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (e.g., from browsers)

# Load the trained model and vectorizer
model = joblib.load("waf_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def preprocess_input(text):
    text = text.lower()
    text = re.sub(r"[^\w\s\'\"<>=/]", "", text)
    return text

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "input" not in data:
        return jsonify({"error": "Missing 'input' field"}), 400

    raw_text = data["input"]
    cleaned = preprocess_input(raw_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized).max()

    return jsonify({
        "input": raw_text,
        "prediction": prediction,
        "confidence": float(round(confidence, 4))
    })

@app.route("/", methods=["GET"])
def serve_demo():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(debug=True)
