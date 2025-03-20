from flask import Flask, request, jsonify
import pandas as pd
from textblob import TextBlob
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


def analyze_sentiment(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity  # Ranges from -1 to 1
    offensiveness_score = (1 - sentiment_score) * 50  # Flip & scale to 0-100
    return round(offensiveness_score, 2)  # Ensure valid range
  # Cap the value at 100%

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    content = data.get("content", "")
    
    if not content:
        return jsonify({"error": "No input provided."}), 400
    
    offensiveness_score = analyze_sentiment(content)
    response = {
        "offensiveness": round(offensiveness_score, 2),
        "message": "This content is likely offensive." if offensiveness_score > 50 else "This content seems safe."
    }
    
    return jsonify(response)

@app.route('/api/predict-file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are allowed."}), 400
    
    df = pd.read_csv(file)
    if 'text' not in df.columns:
        return jsonify({"error": "CSV must have a 'text' column."}), 400
    
    results = []
    for text in df['text']:
        offensiveness_score = analyze_sentiment(str(text))
        
        results.append({
            "text": text,
            "offensiveness": round(offensiveness_score, 2),
            "message": "This content is likely offensive." if offensiveness_score > 50 else "This content seems safe."
        })
    
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render's provided PORT or default to 5000
    app.run(host='0.0.0.0', port=port, debug=False)  # Bind to 0.0.0.0
