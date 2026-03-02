# Resume Filtering & ATS Demo

A simple Streamlit tool that

- classifies uploaded resumes into job categories (logistic model)
- saves each file under the predicted folder
- enables companies to semantically search resumes by job description
- provides a rich ATS‑style parser with fields, links, projects, skills etc.

## Highlights

- Upload PDF/DOCX resumes, see top‑2 matching categories
- Semantic ranking using SBERT embeddings
- Parser extracts name, contact, experience, internships,
  companies, projects (with description), certifications, and more
- Output presented in a form‑like layout for easy copy/paste



```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

Place training model (`final_resume_classifier_logistic.pkl`) under `model/` before running.
