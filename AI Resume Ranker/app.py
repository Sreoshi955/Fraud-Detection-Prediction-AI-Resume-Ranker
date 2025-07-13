import os
import fitz  # PyMuPDF
import spacy
import pandas as pd
from flask import Flask, request, render_template, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load SpaCy
nlp = spacy.load("en_core_web_sm")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

# Preprocess text
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Compute similarity scores
def compute_similarity(resume_texts, job_desc):
    corpus = [job_desc] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    job_vec = tfidf_matrix[0]
    resume_vecs = tfidf_matrix[1:]
    scores = cosine_similarity(job_vec, resume_vecs)[0]
    return scores

# Rank resumes
def rank_resumes(file_paths, job_desc_text):
    resume_texts = [preprocess(extract_text_from_pdf(path)) for path in file_paths]
    job_desc_processed = preprocess(job_desc_text)
    scores = compute_similarity(resume_texts, job_desc_processed)
    ranked = sorted(zip(file_paths, scores), key=lambda x: x[1], reverse=True)
    return ranked

# Save results to CSV
def save_csv(ranked):
    df = pd.DataFrame(ranked, columns=["Resume Path", "Score"])
    df["Resume"] = df["Resume Path"].apply(lambda x: os.path.basename(x))
    df["Score (%)"] = df["Score"].apply(lambda x: round(x * 100, 2))
    df.drop(columns=["Score", "Resume Path"], inplace=True)
    csv_path = os.path.join("static", "ranked_resumes.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    csv_path = None

    if request.method == 'POST':
        job_desc = request.form['job_description']
        files = request.files.getlist('resumes')

        if not job_desc or not files:
            return render_template("index.html", error="Please provide both job description and resumes.")

        file_paths = []
        for f in files:
            save_path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(save_path)
            file_paths.append(save_path)

        ranked = rank_resumes(file_paths, job_desc)
        csv_path = save_csv(ranked)
        results = [(os.path.basename(path), round(score * 100, 2)) for path, score in ranked]

    return render_template("index.html", results=results, csv_path=csv_path)

@app.route('/download')
def download_csv():
    return send_file("static/ranked_resumes.csv", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
