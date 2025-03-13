from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import fitz  # PyMuPDF for PDFs
import newspaper
import os

app = Flask(__name__)
CORS(app)

# Load Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 📌 **Summarization from Text**
@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
    return jsonify({"summary": summary[0]["summary_text"]})

# 📌 **Summarization from News URL**
@app.route("/summarize-url", methods=["POST"])
def summarize_url():
    data = request.json
    url = data.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        summary = summarizer(article.text, max_length=200, min_length=50, do_sample=False)
        return jsonify({"summary": summary[0]["summary_text"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 📌 **Summarization from PDF**
@app.route("/summarize-pdf", methods=["POST"])
def summarize_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    text = extract_text_from_pdf(filepath)
    if not text:
        return jsonify({"error": "Failed to extract text"}), 400

    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
    return jsonify({"summary": summary[0]["summary_text"]})

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if __name__ == "__main__":
    app.run(debug=True)
