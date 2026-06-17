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

## Getting started

1. Create & activate a virtual environment

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Windows (cmd)
.\.venv\Scripts\activate.bat
# Git Bash / WSL
source .venv/Scripts/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Run the app

```bash
streamlit run app.py
```

## Notes about repository contents

- For size and privacy reasons, large model files and raw datasets are not included in this repository. The following paths are gitignored by default:
  - `distilbert_resume_model/`, `model/`, `embeddings/`, `/data/`, `/resumes/`, and `*.safetensors`/`*.pt` files.
- If you have local model weights or datasets, keep them locally in the same folder structure (they are intentionally untracked). To add large files to the repo, consider using Git LFS (`git lfs install` then `git lfs track "*.safetensors"`).

## Environment & secrets

- Create a `.env` file for any secrets (API keys, credentials). `.env` is in `.gitignore` and will not be pushed.

## Where things live

- Main app: `app.py`
- Inference helpers: `distilbert_inference.py`
- ESCO integration: `esco_integration.py`
- Explainers: `shap_explainer.py`
- Parser utilities: `utils/parser.py`

## Contributing

- Open an issue or PR if you'd like to add documentation for a missing model or dataset, or to add sample data (small anonymized examples only).

## License

- (Add license info here if applicable)
