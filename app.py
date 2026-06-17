

import streamlit as st
import os
import joblib
import re
import warnings
import logging
import spacy
import fitz
import numpy as np

from utils.parser import extract_text_from_pdf, extract_text_from_docx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# silence noisy logs
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from esco_integration import extract_esco_skills, combined_score
from shap_explainer import explain_prediction_fast, render_shap_html


# Cached model loaders


@st.cache_resource
def load_tfidf_model():
    return joblib.load("model/final_resume_classifier_logistic.pkl")

@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_distilbert():
    model_dir = "distilbert_resume_model"
    if not os.path.isdir(model_dir):
        return None
    try:
        from distilbert_inference import DistilBertClassifier
        return DistilBertClassifier(model_dir)
    except Exception as e:
        st.warning(f"DistilBERT could not load: {e}")
        return None

tfidf_model  = load_tfidf_model()
sbert_model  = load_sbert()
nlp          = load_spacy()

UPLOAD_DIR    = "resumes"
EMBEDDING_DIR = "embeddings"
os.makedirs(UPLOAD_DIR,    exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)


# Page setup & navigation


st.set_page_config(page_title="Resume Classifier & ATS", page_icon="📄", layout="wide")
st.title("📄 Resume Classifier & ATS System")

menu   = ["User Upload", "Company Search", "Resume Parser"]
choice = st.sidebar.selectbox("Navigation", menu)

# Classifier settings — ONLY show on User Upload page
classifier_choice = "TF-IDF + Logistic Regression"   # default

if choice == "User Upload":
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Classifier Settings")
    classifier_choice = st.sidebar.radio(
        "Choose Classifier",
        ["TF-IDF + Logistic Regression", "DistilBERT"],
        index=0,
        help=(
            "TF-IDF + LR: fast, runs locally, shows SHAP explanation.\n\n"
            "DistilBERT: needs distilbert_resume_model/ folder. "
            "Train it on Google Colab first (train_distilbert_your_dataset.py)."
        )
    )
    st.sidebar.markdown(
        "<small style='color:#888;'>**Note:** 99% accuracy on training data "
        "reflects template memorisation. Confidence may be lower on "
        "human-written, multi-domain resumes.</small>",
        unsafe_allow_html=True
    )


# Shared helpers


def extract_name(text: str) -> str:
    snippet = text[:300]
    doc = nlp(snippet)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


def safe_extract_text(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_docx(uploaded_file)
        if not text or not text.strip():
            st.error("The file appears empty or unreadable. Try a different file.")
            return None
        return text
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return None



# PAGE 1 – User Upload

if choice == "User Upload":

    st.header("📤 Upload Resume")
    st.caption(
        "Upload a PDF or DOCX resume. The system predicts the top 2 matching "
        "job categories and shows why (SHAP) and which skills were detected (ESCO)."
    )

    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

    if uploaded_file:

        text = safe_extract_text(uploaded_file)
        if text is None:
            st.stop()

        cleaned = re.sub(r'[^\w\s]', '', text.lower())

# Classify
        if classifier_choice == "DistilBERT":
            bert_clf = load_distilbert()
            if bert_clf is None:
                st.warning(
                    "⚠️ **distilbert_resume_model/** folder not found. "
                    "Falling back to TF-IDF + LR.\n\n"
                    "To use DistilBERT: run `train_distilbert_your_dataset.py` "
                    "on Google Colab (T4 GPU), download the zip, extract and place "
                    "the folder next to app.py."
                )
                probs       = tfidf_model.predict_proba([cleaned])[0]
                classes     = tfidf_model.classes_
                top_idxs    = probs.argsort()[::-1][:2]
                top_matches = [(classes[i], probs[i]) for i in top_idxs]
                used_model  = "TF-IDF + LR (fallback)"
            else:
                top_matches = bert_clf.predict_top2(text)
                used_model  = "DistilBERT"
        else:
            probs       = tfidf_model.predict_proba([cleaned])[0]
            classes     = tfidf_model.classes_
            top_idxs    = probs.argsort()[::-1][:2]
            top_matches = [(classes[i], probs[i]) for i in top_idxs]
            used_model  = "TF-IDF + LR"

        category = top_matches[0][0]

# Save resume file + SBERT embedding
        cat_folder = os.path.join(UPLOAD_DIR, category)
        os.makedirs(cat_folder, exist_ok=True)
        with open(os.path.join(cat_folder, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        embedding = sbert_model.encode([text])[0]
        joblib.dump(embedding, os.path.join(EMBEDDING_DIR, uploaded_file.name + ".pkl"))

# Results
        st.markdown("---")
        st.subheader("🎯 Classification Result")

        conf1 = top_matches[0][1] * 100
        conf2 = top_matches[1][1] * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Model Used", used_model)
        col2.metric(f"1st: {top_matches[0][0]}", f"{conf1:.1f}%")
        col3.metric(f"2nd: {top_matches[1][0]}", f"{conf2:.1f}%")

        st.success(f"✅ Resume saved under category **{category}**")

# Low confidence warning (< 50
        if conf1 < 50:
            st.info(
                f"💡 **Confidence is {conf1:.1f}%** — lower than usual. "
                "This happens when a resume spans multiple domains "
                "(e.g. Python + ML + Web Development together). "
                "The model was trained on single-domain template resumes, "
                "so multi-domain or academic-format resumes produce lower confidence. "
                "The top prediction is still the best match available."
            )

# SHAP explanation
        st.markdown("---")
        if "TF-IDF" in used_model:
            with st.expander("🔍 Why this prediction? (Feature Contribution Analysis)"):
                st.caption(
                    "Each word's contribution = TF-IDF weight × LR coefficient. "
                    "Blue bars push the model **toward** this category. "
                    "Red bars push it **away**."
                )
                try:
                    expl = explain_prediction_fast(
                        tfidf_model, cleaned, category, top_n=15
                    )
                    html = render_shap_html(expl)
                    st.components.v1.html(html, height=450, scrolling=True)
                except Exception as e:
                    st.warning(f"Explanation could not be generated: {e}")
        else:
            st.info(
                "💡 Feature explanations are only available for TF-IDF + LR. "
                "Switch the classifier in the sidebar to see them."
            )

# ESCO skill badges
        st.markdown("---")
        with st.expander("🛠️ ESCO Skills Detected in this Resume", expanded=True):
            st.caption(
                "Skills matched against ESCO canonical dictionary. "
                "e.g. 'sklearn' → 'Scikit-learn', 'cnn' → 'Convolutional Neural Network', "
                "'sbert' → 'Sentence-BERT'."
            )
            esco_skills = extract_esco_skills(text)

            if esco_skills:
                def skill_color(skill):
                    lang_skills = {
                        "Python", "Java", "JavaScript", "TypeScript", "Go",
                        "Kotlin", "SQL", "R (programming language)", "Bash scripting",
                        "C (programming language)", "C#", "Rust", "Scala", "PHP",
                        "Dart", "C++", "Swift"
                    }
                    cloud_skills = {
                        "Amazon Web Services", "Microsoft Azure", "Google Cloud Platform",
                        "Docker", "Kubernetes", "Terraform", "CI/CD", "Jenkins",
                        "GitHub Actions", "GitLab CI", "Helm", "ArgoCD", "Ansible"
                    }
                    ml_skills = {
                        "TensorFlow", "PyTorch", "Scikit-learn", "Keras", "XGBoost",
                        "Convolutional Neural Network", "Natural language processing",
                        "Computer vision", "Sentence-BERT", "Hugging Face",
                        "OpenCV", "spaCy", "Logistic Regression", "TF-IDF",
                        "Cosine Similarity", "FastAI", "YOLO", "MLflow", "Kubeflow"
                    }
                    if skill in lang_skills:
                        return "#2E7D32"    # green – languages
                    elif skill in cloud_skills:
                        return "#6A1B9A"   # purple – cloud/devops
                    elif skill in ml_skills:
                        return "#E65100"   # orange – ML/AI
                    else:
                        return "#1565C0"   # blue – frameworks/tools

                badges = " ".join(
                    f'<span style="background:{skill_color(s)};color:white;'
                    f'padding:5px 12px;border-radius:16px;margin:4px;'
                    f'display:inline-block;font-size:13px;font-weight:500;">{s}</span>'
                    for s in esco_skills
                )
# Legend
                legend = (
                    '<div style="font-size:12px;color:#888;margin-bottom:8px;">'
                    '<span style="background:#2E7D32;color:white;padding:2px 8px;border-radius:8px;margin-right:6px;">Languages</span>'
                    '<span style="background:#E65100;color:white;padding:2px 8px;border-radius:8px;margin-right:6px;">ML / AI</span>'
                    '<span style="background:#6A1B9A;color:white;padding:2px 8px;border-radius:8px;margin-right:6px;">Cloud / DevOps</span>'
                    '<span style="background:#1565C0;color:white;padding:2px 8px;border-radius:8px;">Frameworks / Tools</span>'
                    '</div>'
                )
                st.components.v1.html(
                    f'<div style="font-family:sans-serif;">{legend}'
                    f'<div style="line-height:2.4;">{badges}</div></div>',
                    height=max(120, len(esco_skills) * 12),
                    scrolling=True
                )
                st.caption(f"Total: **{len(esco_skills)}** ESCO skills detected.")
            else:
                st.info("No ESCO-recognised skills detected in this resume.")



elif choice == "Company Search":

    st.header("🔍 Semantic Resume Search")
    st.caption(
        "Enter a job description. Resumes are ranked by: "
        "**SBERT semantic similarity (65%) + ESCO skill overlap (35%)**."
    )

    if not os.listdir(UPLOAD_DIR):
        st.warning("No resumes uploaded yet. Go to **User Upload** first.")
        st.stop()

    categories  = sorted(os.listdir(UPLOAD_DIR))
    selected    = st.selectbox("Choose Category", categories)
    job_description = st.text_area(
        "Enter Job Description",
        height=200,
        placeholder=(
            "e.g. We are looking for a Python Developer with Django, FastAPI, "
            "PostgreSQL, Redis, Docker experience..."
        )
    )

    if st.button("🔎 Search Resumes"):

        if not job_description.strip():
            st.warning("Please enter a job description.")
            st.stop()

        folder = os.path.join(UPLOAD_DIR, selected)
        if not os.path.exists(folder) or not os.listdir(folder):
            st.warning(f"No resumes in **{selected}**.")
            st.stop()

        resumes       = os.listdir(folder)
        job_embedding = sbert_model.encode([job_description])[0]
        jd_esco       = extract_esco_skills(job_description)
        resume_scores = []

        with st.spinner("Ranking resumes..."):
            for resume in resumes:
                file_path      = os.path.join(folder, resume)
                embedding_path = os.path.join(EMBEDDING_DIR, resume + ".pkl")
                if not os.path.exists(embedding_path):
                    continue

                resume_embedding = joblib.load(embedding_path)
                sbert_sim = float(
                    cosine_similarity([job_embedding], [resume_embedding])[0][0]
                )

                try:
                    if resume.lower().endswith(".pdf"):
                        with open(file_path, "rb") as f:
                            res_text = extract_text_from_pdf(f)
                    elif resume.lower().endswith(".docx"):
                        res_text = extract_text_from_docx(file_path)
                    else:
                        res_text = ""
                except Exception:
                    res_text = ""

                final_sim, r_skills, _, esco_s = combined_score(
                    sbert_sim, res_text, job_description,
                    sbert_weight=0.65, esco_weight=0.35
                )

                matched = sorted(set(r_skills) & set(jd_esco))
                missing = sorted(set(jd_esco) - set(r_skills))

                resume_scores.append({
                    "name":    resume,
                    "final":   final_sim,
                    "sbert":   sbert_sim,
                    "esco":    esco_s,
                    "text":    res_text,
                    "matched": matched,
                    "missing": missing,
                })

        if not resume_scores:
            st.warning("No embeddings found. Re-upload resumes via User Upload.")
            st.stop()

        ranked = sorted(resume_scores, key=lambda x: x["final"], reverse=True)
        st.markdown(f"### Results — {len(ranked)} resume(s) ranked")

        for i, r in enumerate(ranked):
            label = (
                f"Rank {i+1}: {r['name']}  |  "
                f"Combined: {r['final']*100:.1f}%  |  "
                f"SBERT: {r['sbert']*100:.1f}%  |  "
                f"ESCO Overlap: {r['esco']*100:.1f}%"
            )
            with st.expander(label):
                c1, c2, c3 = st.columns(3)
                c1.metric("Combined Score",   f"{r['final']*100:.1f}%")
                c2.metric("SBERT Similarity", f"{r['sbert']*100:.1f}%")
                c3.metric("ESCO Overlap",     f"{r['esco']*100:.1f}%")

                if r["matched"]:
                    st.markdown("**✅ Skills matched:** " + " · ".join(r["matched"]))
                if r["missing"]:
                    st.markdown("**❌ Skills in JD but missing from resume:** " +
                                " · ".join(r["missing"]))

                st.text_area(
                    "Resume Content",
                    r["text"] or "[Could not read]",
                    height=280,
                    key=f"res_{i}"
                )



# PAGE 3 – Resume Parser


elif choice == "Resume Parser":

    st.header("🗂️ Professional ATS Resume Parser")
    st.caption(
        "Upload a resume to extract structured fields. "
        "Skills use ESCO canonical names (32+ ML/AI/web skills supported)."
    )

    def extract_pdf_links(file):
        links_found = []
        try:
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            for page in pdf:
                for link in page.get_links():
                    if "uri" in link:
                        links_found.append(link["uri"])
            pdf.close()
        except Exception:
            pass
        return links_found

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

    if uploaded_file:

        text = safe_extract_text(uploaded_file)
        if text is None:
            st.stop()

# Links
        raw_links = set()
        if uploaded_file.type == "application/pdf":
            uploaded_file.seek(0)
            for link in extract_pdf_links(uploaded_file):
                raw_links.add(link.strip())
            uploaded_file.seek(0)
        for url in re.findall(r'https?://\S+', text):
            raw_links.add(url.strip())

        linkedin = ""
        github   = ""
        other_links = []
        for link in raw_links:
            ll = link.lower()
            if "linkedin.com" in ll:
                linkedin = link
            elif "github.com" in ll:
                github = link
            else:
                other_links.append(link)

        lower_text = text.lower()
        for kw in ["leetcode", "geeksforgeeks", "gfg", "kaggle",
                   "hackerrank", "codechef", "portfolio"]:
            if kw in lower_text and kw.capitalize() not in other_links:
                other_links.append(kw.capitalize())

        if not linkedin and "linkedin" in lower_text:
            linkedin = "Mentioned (no URL found)"
        if not github and "github" in lower_text:
            github = "Mentioned (no URL found)"

# Basic fields
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        name  = extract_name(text)

        email = (re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text) or [""])[0]
        phone = (re.findall(r'\+?\d[\d\s-]{8,15}\d', text) or [""])[0]

        cgpa = ""
        m = re.search(r'(cgpa|gpa|cpi)\s*[:\-–]?\s*(\d+\.?\d*)', lower_text)
        if m:
            cgpa = m.group(2)

# Section splitting
        HEADERS = [
            "education", "experience", "projects", "technical skills",
            "skills", "achievements", "certifications", "internships",
            "summary", "objective", "profile", "position of responsibility"
        ]
        sections: dict = {}
        current = "general"
        sections[current] = []
        for line in lines:
            h = line.lower().strip()
            matched = next((hdr for hdr in HEADERS if h == hdr or h.startswith(hdr)), None)
            if matched:
                current = matched
                sections.setdefault(current, [])
            else:
                sections.setdefault(current, []).append(line)

# Education
        degree = university = ""
        edu_text = " ".join(sections.get("education", []))
        for d in ["b.tech", "bachelor", "m.tech", "mba", "b.sc", "m.sc",
                  "phd", "b.e", "m.e", "bca", "mca"]:
            if d in edu_text.lower():
                degree = d.upper()
                break
        edu_doc = nlp(edu_text[:500])
        for ent in edu_doc.ents:
            if ent.label_ == "ORG":
                university = ent.text
                break

# Experience
        companies    = []
        exp_text     = " ".join(sections.get("experience", []))
        exp_doc      = nlp(exp_text[:1000])
        for ent in exp_doc.ents:
            if ent.label_ == "ORG":
                companies.append(ent.text)
        companies = list(set(companies))

        experience = ""
        m2 = re.findall(r'(\d+)\+?\s+years?\s+of\s+experience', lower_text)
        if m2:
            experience = m2[0] + " years"

        internships = [l for l in lines if "intern" in l.lower()]

# Projects
        projects = []
        cur_proj = None
        for line in sections.get("projects", []):
            if not line.startswith(("•", "-", "–", "*")) and len(line) > 4:
                if cur_proj:
                    projects.append(cur_proj)
                cur_proj = {"title": line.strip(), "description": ""}
            elif cur_proj:
                cur_proj["description"] += line.strip() + " "
        if cur_proj:
            projects.append(cur_proj)

# Skills – ESCO
        esco_skills = extract_esco_skills(text)

        raw_skills = []
        for line in sections.get("technical skills", []) + sections.get("skills", []):
            if ":" in line:
                part = line.split(":", 1)[1]
                raw_skills.extend([s.strip() for s in part.split(",")])

        esco_lower  = {s.lower() for s in esco_skills}
        extra_raw   = [s for s in raw_skills if s and s.lower() not in esco_lower]
        final_skills = esco_skills + extra_raw

# Certifications
        cert_lines = (sections.get("achievements", []) +
                      sections.get("certifications", []))
        certifications = [
            l for l in cert_lines
            if any(kw in l.lower() for kw in [
                "certified", "certification", "certificate",
                "aws", "gcp", "azure", "oracle", "microsoft",
                "google", "coursera", "udemy", "istqb", "teachnook"
            ])
        ]

# Display
        st.markdown("---")
        st.subheader("📋 Structured Resume Profile")

        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Name",           name,       key="p_name")
            st.text_input("Email",          email,      key="p_email")
            st.text_input("Phone",          phone,      key="p_phone")
            st.text_input("Degree",         degree,     key="p_deg")
            st.text_input("University",     university, key="p_uni")
            st.text_input("CGPA / GPA",     cgpa,       key="p_cgpa")
        with col2:
            st.text_input("LinkedIn",       linkedin,   key="p_li")
            st.text_input("GitHub",         github,     key="p_gh")
            st.text_area("Other Links",     "\n".join(other_links), height=80, key="p_lnk")
            st.text_area("Companies",       "\n".join(companies),   height=80, key="p_co")
            st.text_input("Experience",     experience, key="p_exp")

        st.text_area("Internships",
                     "\n".join(internships) or "Not mentioned",
                     height=80, key="p_int")

        proj_lines = []
        for p in projects:
            line = p["title"]
            if p["description"].strip():
                line += ": " + p["description"].strip()
            proj_lines.append(line)
        st.text_area("Projects",
                     "\n\n".join(proj_lines) or "Not mentioned",
                     height=200, key="p_prj")

        st.text_area("Certifications",
                     "\n".join(certifications) or "Not mentioned",
                     height=80, key="p_cert")

# ESCO skill badges
        st.markdown("**🛠️ Skills (ESCO Canonical Names)**")
        st.caption(
            "Matched against ESCO skill dictionary. "
            "Green = languages · Orange = ML/AI · Purple = Cloud/DevOps · Blue = frameworks/tools."
        )

        if final_skills:
            def skill_color(skill):
                lang_set  = {"Python","Java","JavaScript","TypeScript","Go","Kotlin","SQL",
                             "R (programming language)","Bash scripting","C (programming language)",
                             "C#","Rust","Scala","PHP","Dart","C++","Swift","HTML","CSS"}
                cloud_set = {"Amazon Web Services","Microsoft Azure","Google Cloud Platform",
                             "Docker","Kubernetes","Terraform","CI/CD","Jenkins",
                             "GitHub Actions","GitLab CI","Helm","ArgoCD","Ansible"}
                ml_set    = {"TensorFlow","PyTorch","Scikit-learn","Keras","XGBoost",
                             "Convolutional Neural Network","Natural language processing",
                             "Computer vision","Sentence-BERT","Hugging Face","OpenCV",
                             "spaCy","Logistic Regression","TF-IDF","Cosine Similarity",
                             "FastAI","YOLO","MLflow","Kubeflow","NumPy","Pandas"}
                if skill in lang_set:  return "#2E7D32"
                if skill in cloud_set: return "#6A1B9A"
                if skill in ml_set:    return "#E65100"
                return "#1565C0"

            badges = " ".join(
                f'<span style="background:{skill_color(s)};color:white;'
                f'padding:5px 13px;border-radius:16px;margin:4px;'
                f'display:inline-block;font-size:13px;font-weight:500;">{s}</span>'
                for s in final_skills
            )
            legend = (
                '<div style="font-size:12px;color:#aaa;margin-bottom:10px;">'
                '<span style="background:#2E7D32;color:white;padding:2px 8px;'
                'border-radius:8px;margin-right:6px;">Languages</span>'
                '<span style="background:#E65100;color:white;padding:2px 8px;'
                'border-radius:8px;margin-right:6px;">ML / AI</span>'
                '<span style="background:#6A1B9A;color:white;padding:2px 8px;'
                'border-radius:8px;margin-right:6px;">Cloud / DevOps</span>'
                '<span style="background:#1565C0;color:white;padding:2px 8px;'
                'border-radius:8px;">Frameworks / Tools</span></div>'
            )
            st.components.v1.html(
                f'<div style="font-family:sans-serif;">{legend}'
                f'<div style="line-height:2.5;">{badges}</div></div>',
                height=max(130, len(final_skills) * 11),
                scrolling=True
            )
            st.caption(f"**{len(final_skills)}** skills detected.")
        else:
            st.info("No ESCO-recognised skills detected.")

        with st.expander("📄 Skills as plain text (copy-paste)"):
            st.text_area("Detected skills", ", ".join(final_skills), height=80, key="p_plain", label_visibility="collapsed")