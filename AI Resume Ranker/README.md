AI Resume Ranker — Flask Web App

An AI-powered web application to automatically rank resumes against a given job description using Natural Language Processing (NLP) techniques. Built with a clean and intuitive Flask-based interface, this tool streamlines resume screening and helps recruiters and hiring managers shortlist candidates efficiently.

Features

🔹 Extracts and processes resume content (PDFs) using PyMuPDF
🔹 Compares resumes against a provided job description using TF-IDF + Cosine Similarity.
🔹 Ranks candidates based on relevance scores
🔹 Generates a downloadable CSV report for easy review
🔹 Sleek and responsive web interface (Flask + HTML) for uploading resumes and entering job descriptions
🔹 Secure and easy-to-use, designed for real-world usage in HRTech and recruitment automation

Tech Stack

 Python = Backend and core logic                       
 Flask = Web framework for UI and API                 
 SpaCy = NLP pipeline for text extraction and parsing 
 PyMuPDF (fitz) = PDF parsing and resume content extraction    
 scikit-learn = TF-IDF vectorization and similarity scoring  
 HTML/CSS = Frontend styling and layout                  

Project Structure

ai-resume-ranker/
│
├── static/
│   └── logo.jpeg               # For using logo
│
├── templates/
│   └── index.html               # UI for resume upload and job description input
│
├── uploads/                    # Stores uploaded resumes temporarily
│
├── app.py                      # Flask app with routing and logic
├── resume_ranker.py            # Core ranking logic using NLP
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

Output

* CSV file containing ranked resumes with similarity scores
* Preview of top matches directly on the web interface

Ideal For

* HR professionals or recruiters
* Startups automating hiring workflows
* Developers experimenting with **AI + NLP** in real use cases
* Anyone who wants to explore document parsing and text similarity

Learning Highlights

This project was an exciting opportunity to apply NLP and web development to solve a real-world HR challenge. I explored:

* Resume parsing with PyMuPDF
* Semantic text matching using TF-IDF and cosine similarity
* Integrating AI logic with a Flask web frontend
* CSV report generation and file handling.


