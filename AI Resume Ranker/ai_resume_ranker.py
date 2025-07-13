import os
import fitz  # PyMuPDF
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Create folders if not exist
RESUME_FOLDER = "resumes"
JOB_DESC_FILE = "job_description.txt"
OUTPUT_FILE = "ranked_resumes.csv"

os.makedirs(RESUME_FOLDER, exist_ok=True)

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Preprocess text using SpaCy
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Step 3: Compute similarity using TF-IDF
def compute_similarity(resume_texts, job_description_text):
    corpus = [job_description_text] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    job_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(job_vector, resume_vectors)[0]
    return scores

# Step 4: Rank resumes
def rank_resumes(resume_folder, job_description_text):
    resume_files = [os.path.join(resume_folder, f) for f in os.listdir(resume_folder) if f.endswith(".pdf")]
    
    if not resume_files:
        print("⚠️ No PDF resumes found in the 'resumes/' folder.")
        return []

    resume_texts = [preprocess(extract_text_from_pdf(f)) for f in resume_files]
    scores = compute_similarity(resume_texts, preprocess(job_description_text))
    ranked = sorted(zip(resume_files, scores), key=lambda x: x[1], reverse=True)
    return ranked

# Step 5: Export results
def export_to_csv(ranked_list, output_csv):
    df = pd.DataFrame(ranked_list, columns=["Resume Path", "Score"])
    df["Resume"] = df["Resume Path"].apply(lambda x: os.path.basename(x))
    df["Score (%)"] = df["Score"].apply(lambda x: round(x * 100, 2))
    df.drop(columns="Score", inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Ranked results saved to: {output_csv}")

# Main
if __name__ == "__main__":
    print("🔍 AI Resume Ranker Started...")

    # Step 0: Load job description
    if not os.path.exists(JOB_DESC_FILE):
        print(f"⚠️ Job description file '{JOB_DESC_FILE}' not found.")
        print("👉 Please create a 'job_description.txt' file with the job requirements.")
        exit()

    with open(JOB_DESC_FILE, "r", encoding="utf-8") as f:
        job_description_text = f.read().strip()

    # Rank resumes
    ranked_resumes = rank_resumes(RESUME_FOLDER, job_description_text)

    if ranked_resumes:
        # Print results
        print("\n📋 Ranked Resumes:")
        for idx, (filename, score) in enumerate(ranked_resumes, start=1):
            print(f"{idx}. {os.path.basename(filename)} — Score: {score * 100:.2f}%")

        # Export CSV
        export_to_csv(ranked_resumes, OUTPUT_FILE)
    else:
        print("❌ No resumes ranked.")