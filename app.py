import streamlit as st
import os
import joblib
import re
import warnings
import logging
import spacy
import fitz
from utils.parser import extract_text_from_pdf, extract_text_from_docx
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity


try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Load models

model = joblib.load("model/final_resume_classifier_logistic.pkl")

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

UPLOAD_DIR = "resumes"
EMBEDDING_DIR = "embeddings"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

st.title("📄 Resume Classifier & ATS System")

menu = ["User Upload", "Company Search", "Resume Parser"]
choice = st.sidebar.selectbox("Navigation", menu)

# USER UPLOAD

if choice == "User Upload":

    st.header("Upload Resume")

    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

    if uploaded_file:

        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_docx(uploaded_file)

        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        # compute probabilities using classifier directly (no label encoder)
        probs = model.predict_proba([cleaned])[0]
        classes = model.classes_
        # extract top two predictions
        top_idxs = probs.argsort()[::-1][:2]
        top_matches = [(classes[i], probs[i]) for i in top_idxs]
        category = top_matches[0][0]

        category_folder = os.path.join(UPLOAD_DIR, category)
        os.makedirs(category_folder, exist_ok=True)
        save_path = os.path.join(category_folder, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        embedding = sbert_model.encode([text])[0]
        joblib.dump(embedding, os.path.join(EMBEDDING_DIR, uploaded_file.name + ".pkl"))

        st.success(
            f"Top matches: 1. {top_matches[0][0]} ({top_matches[0][1]*100:.2f}%), "
            f"2. {top_matches[1][0]} ({top_matches[1][1]*100:.2f}%)"
        )

# COMPANY SEARCH

elif choice == "Company Search":

    st.header("Semantic Resume Search")

    categories = os.listdir(UPLOAD_DIR)
    selected = st.selectbox("Choose Category", categories)
    job_description = st.text_area("Enter Job Description")

    if st.button("Search"):

        folder = os.path.join(UPLOAD_DIR, selected)

        if os.path.exists(folder):

            resumes = os.listdir(folder)
            job_embedding = sbert_model.encode([job_description])[0]
            resume_scores = []

            for resume in resumes:

                file_path = os.path.join(folder, resume)
                embedding_path = os.path.join(EMBEDDING_DIR, resume + ".pkl")

                if not os.path.exists(embedding_path):
                    continue

                resume_embedding = joblib.load(embedding_path)
                similarity = cosine_similarity(
                    [job_embedding], [resume_embedding]
                )[0][0]

                if resume.lower().endswith(".pdf"):
                    with open(file_path, "rb") as f:
                        text = extract_text_from_pdf(f)
                elif resume.lower().endswith(".docx"):
                    text = extract_text_from_docx(file_path)
                else:
                    text = ""

                resume_scores.append((resume, similarity, text))

            ranked = sorted(resume_scores, key=lambda x: x[1], reverse=True)

            for i, (resume, score, text) in enumerate(ranked):
                with st.expander(f"Rank {i+1}: {resume} | Match: {round(score*100,2)}%"):
                    # provide a unique key so Streamlit doesn't complain about duplicate IDs
                    st.text_area("Resume Content", text, height=300, key=f"resume_{i}")

# ADVANCED ATS RESUME PARSER

elif choice == "Resume Parser":

    st.header("Professional ATS Resume Parser")

    def extract_pdf_links(file):
        links_found = []
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            for link in page.get_links():
                if "uri" in link:
                    links_found.append(link["uri"])
        pdf.close()
        return links_found

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

    if uploaded_file:

        linkedin = ""
        github = ""
        other_links = []

        if uploaded_file.type == "application/pdf":
            links = extract_pdf_links(uploaded_file)
            uploaded_file.seek(0)
            text = extract_text_from_pdf(uploaded_file)

            for link in links:
                link_lower = link.lower()
                if "linkedin.com" in link_lower:
                    linkedin = link
                elif "github.com" in link_lower:
                    github = link
                else:
                    other_links.append(link)
        else:
            text = extract_text_from_docx(uploaded_file)

        lower_text = text.lower()
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        name = lines[0] if lines else ""

        email_match = re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text)
        email = email_match[0] if email_match else ""

        phone_match = re.findall(r'\+?\d[\d\s-]{8,15}\d', text)
        phone = phone_match[0] if phone_match else ""

        cgpa = ""
        cgpa_match = re.search(r'(cgpa|gpa|cpi)\s*[:\-–]?\s*(\d+\.?\d*)', lower_text)
        if cgpa_match:
            cgpa = cgpa_match.group(2)

        sections = {}
        current = "general"
        sections[current] = []

        for line in lines:
            header = line.lower()
            if any(h in header for h in
                   ["education", "experience", "projects",
                    "technical skills", "skills",
                    "achievements", "certifications"]):
                current = header
                sections[current] = []
            else:
                sections[current].append(line)

        degree = ""
        university = ""

        education_text = " ".join(sections.get("education", []))

        degree_keywords = ["b.tech", "bachelor", "m.tech", "mba",
                           "b.sc", "m.sc", "phd", "b.e", "m.e"]

        for d in degree_keywords:
            if d in education_text.lower():
                degree = d
                break

        edu_doc = nlp(education_text)
        for ent in edu_doc.ents:
            if ent.label_ == "ORG":
                university = ent.text
                break

        companies = []
        experience_text = " ".join(sections.get("experience", []))
        exp_doc = nlp(experience_text)

        for ent in exp_doc.ents:
            if ent.label_ == "ORG":
                companies.append(ent.text)

        companies = list(set(companies))

        experience = ""
        exp_match = re.findall(r'(\d+)\+?\s+years? of experience', lower_text)
        if exp_match:
            experience = exp_match[0] + " years"

        internships = []
        for line in lines:
            if "intern" in line.lower():
                internships.append(line)

        projects = []
        project_section = sections.get("projects", [])
        current_project = None

        for line in project_section:
            if not line.startswith("•") and len(line) > 3:
                if current_project:
                    projects.append(current_project)
                current_project = {"title": line.strip(), "description": ""}
            elif current_project:
                current_project["description"] += line.strip() + " "

        if current_project:
            projects.append(current_project)

        url_pattern = r'https?://\S+'
        all_urls = re.findall(url_pattern, text)

        for link in all_urls:
            link_lower = link.lower()
            if "linkedin.com" in link_lower:
                linkedin = link
            elif "github.com" in link_lower:
                github = link
            else:
                other_links.append(link)

        platform_keywords = [
            "github", "linkedin", "leetcode",
            "geeksforgeeks", "gfg",
            "kaggle", "hackerrank",
            "codechef", "portfolio"
        ]

        for keyword in platform_keywords:
            if keyword in lower_text:
                if keyword == "linkedin" and not linkedin:
                    linkedin = "Mentioned"
                elif keyword == "github" and not github:
                    github = "Mentioned"
                else:
                    other_links.append(keyword.capitalize())

        other_links = list(set(other_links))

        skills = []
        for line in sections.get("technical skills", []) + sections.get("skills", []):
            if ":" in line:
                skill_part = line.split(":")[1]
                skills.extend([s.strip() for s in skill_part.split(",")])

        final_skills = list(set(skills))

        certifications = []
        for line in sections.get("achievements", []) + sections.get("certifications", []):
            if "certified" in line.lower():
                certifications.append(line)

        st.subheader("Structured Resume Profile")

        with st.form("resume_output_form"):

            col1, col2 = st.columns(2)

            with col1:
                st.text_input("Name", name)
                st.text_input("Email", email)
                st.text_input("Phone", phone)
                st.text_input("Degree", degree)
                st.text_input("University", university)
                st.text_input("CGPA", cgpa)

            with col2:
                st.text_input("LinkedIn", linkedin)
                st.text_input("GitHub", github)
                st.text_area("Other Links", "\n".join(other_links), height=100)
                st.text_area("Companies", "\n".join(companies), height=100)
                st.text_input("Experience (Years)", experience)

            st.text_area("Internships", "\n".join(internships), height=120)

            project_lines = []
            for proj in projects:
                line = proj["title"]
                if proj["description"]:
                    line += ": " + proj["description"]
                project_lines.append(line)

            st.text_area("Projects", "\n\n".join(project_lines), height=200)

            st.text_area("Certifications", "\n".join(certifications), height=120)

            st.text_area("Skills", ", ".join(final_skills), height=120)

            st.form_submit_button("Done")