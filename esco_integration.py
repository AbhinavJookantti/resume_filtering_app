"""
esco_integration.py
────────────────────
ESCO (European Skills, Competences, Qualifications and Occupations) skill
extraction and overlap scoring.

Two uses:
  1. Resume Parser  – replace colon-split regex with ESCO-validated extraction
  2. Company Search – add ESCO skill overlap score on top of SBERT similarity

No internet connection required at inference time.
The ESCO skill list is embedded directly below as a curated dictionary
covering all 25 resume categories in your dataset.

Usage:
    from esco_integration import extract_esco_skills, esco_overlap_score

    # In Resume Parser
    skills = extract_esco_skills(resume_text)

    # In Company Search (combine with SBERT)
    resume_skills = extract_esco_skills(resume_text)
    jd_skills     = extract_esco_skills(job_description)
    overlap       = esco_overlap_score(resume_skills, jd_skills)
    final_score   = (sbert_score * 0.65) + (overlap * 0.35)
"""

import re
from typing import List, Tuple, Set

# ─────────────────────────────────────────────────────────────────────────────
# ESCO-aligned skill dictionary
# Key   = canonical ESCO skill name
# Value = list of surface forms / aliases that map to this canonical name
# ─────────────────────────────────────────────────────────────────────────────
ESCO_SKILL_MAP: dict[str, list[str]] = {
    # ── Programming languages ────────────────────────────────────────────────
    "Python": ["python", "python3", "py"],
    "Java": ["java", "java8", "java 8", "java11", "core java"],
    "JavaScript": ["javascript", "js", "es6", "ecmascript", "vanilla js"],
    "TypeScript": ["typescript", "ts"],
    "Kotlin": ["kotlin"],
    "Go": ["golang", "go language", "go lang"],
    "SQL": ["sql", "structured query language"],
    "R (programming language)": ["r programming", " r ", "rlang"],
    "Bash scripting": ["bash", "shell scripting", "shell script", "bash scripting"],
    "C++": ["c++", "cpp"],
    "Swift": ["swift", "swiftui"],
    "PHP": ["php"],

    # ── Web & frameworks ──────────────────────────────────────────────────────
    "React": ["react", "reactjs", "react.js"],
    "Vue.js": ["vue", "vuejs", "vue.js"],
    "Angular": ["angular", "angularjs"],
    "Node.js": ["node", "nodejs", "node.js"],
    "Express.js": ["express", "expressjs", "express.js"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi", "fast api"],
    "Spring Boot": ["spring boot", "springboot", "spring-boot"],
    "Spring Framework": ["spring", "spring mvc", "spring security"],
    "Hibernate": ["hibernate", "jpa", "java persistence api"],

    # ── Data & ML ────────────────────────────────────────────────────────────
    "TensorFlow": ["tensorflow", "tf", "tf2"],
    "PyTorch": ["pytorch", "torch"],
    "Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Apache Spark": ["spark", "apache spark", "pyspark"],
    "Apache Kafka": ["kafka", "apache kafka"],
    "Apache Airflow": ["airflow", "apache airflow"],
    "Hadoop": ["hadoop", "hdfs", "mapreduce"],
    "dbt": ["dbt", "data build tool"],
    "MLflow": ["mlflow", "ml flow"],
    "Kubeflow": ["kubeflow"],
    "XGBoost": ["xgboost", "xgb"],
    "Keras": ["keras"],
    "Feature engineering": ["feature engineering"],
    "Natural language processing": ["nlp", "natural language processing", "text mining"],
    "Computer vision": ["computer vision", "cv", "image recognition"],
    "Data analysis": ["data analysis", "data analytics", "exploratory data analysis", "eda"],
    "Data visualisation": ["data visualization", "data visualisation", "matplotlib", "seaborn", "plotly"],
    "Statistics": ["statistics", "statistical analysis", "hypothesis testing", "a/b testing"],

    # ── Cloud & infrastructure ───────────────────────────────────────────────
    "Amazon Web Services": ["aws", "amazon web services", "amazon aws"],
    "Microsoft Azure": ["azure", "microsoft azure"],
    "Google Cloud Platform": ["gcp", "google cloud", "google cloud platform"],
    "Terraform": ["terraform", "hcl"],
    "Docker": ["docker", "containerisation", "containerization"],
    "Kubernetes": ["kubernetes", "k8s", "kubectl"],
    "AWS Lambda": ["lambda", "aws lambda", "serverless"],
    "Amazon S3": ["s3", "amazon s3"],
    "Amazon EC2": ["ec2", "amazon ec2"],
    "Amazon RDS": ["rds", "amazon rds"],
    "CI/CD": ["ci/cd", "continuous integration", "continuous deployment", "continuous delivery"],
    "Jenkins": ["jenkins"],
    "GitHub Actions": ["github actions"],
    "GitLab CI": ["gitlab ci", "gitlab"],
    "Ansible": ["ansible"],
    "Helm": ["helm", "helm charts"],
    "Prometheus": ["prometheus"],
    "Grafana": ["grafana"],
    "ArgoCD": ["argocd", "argo cd"],
    "HashiCorp Vault": ["vault", "hashicorp vault"],

    # ── Databases ─────────────────────────────────────────────────────────────
    "PostgreSQL": ["postgresql", "postgres", "psql"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "Elasticsearch": ["elasticsearch", "elastic search", "elk"],
    "Apache Cassandra": ["cassandra", "apache cassandra"],
    "Snowflake": ["snowflake"],
    "Amazon Redshift": ["redshift", "amazon redshift"],
    "BigQuery": ["bigquery", "google bigquery"],
    "SQLite": ["sqlite"],
    "Oracle Database": ["oracle db", "oracle database", "pl/sql"],
    "Microsoft SQL Server": ["sql server", "mssql", "t-sql", "tsql", "ssrs", "ssis"],

    # ── Mobile ──────────────────────────────────────────────────────────────
    "Android development": ["android", "android sdk", "android development"],
    "Jetpack Compose": ["jetpack compose", "compose"],
    "Firebase": ["firebase"],
    "Retrofit": ["retrofit"],
    "Room (Android)": ["room database", "room db"],
    "Dagger/Hilt": ["dagger", "hilt", "dependency injection"],

    # ── Security ─────────────────────────────────────────────────────────────
    "Network security": ["network security", "firewall", "firewalls", "ids/ips"],
    "Penetration testing": ["penetration testing", "pen testing", "pentest"],
    "SIEM": ["siem", "splunk siem"],
    "Risk assessment": ["risk assessment", "vulnerability assessment"],
    "OWASP": ["owasp"],
    "Incident response": ["incident response", "soc", "security operations"],

    # ── Testing ──────────────────────────────────────────────────────────────
    "Selenium": ["selenium", "selenium webdriver"],
    "Cypress": ["cypress"],
    "Appium": ["appium"],
    "Jest": ["jest"],
    "JUnit": ["junit", "junit5"],
    "TestNG": ["testng"],
    "Pytest": ["pytest"],
    "REST Assured": ["rest assured"],
    "Manual testing": ["manual testing", "test cases", "test case writing"],
    "Regression testing": ["regression testing", "regression test"],
    "BDD": ["bdd", "cucumber", "gherkin", "behaviour driven"],

    # ── Design & UX ──────────────────────────────────────────────────────────
    "Figma": ["figma"],
    "Sketch": ["sketch"],
    "Adobe XD": ["adobe xd", "xd"],
    "User research": ["user research", "usability testing", "user interviews"],
    "Wireframing": ["wireframing", "wireframes", "wireframe"],
    "Prototyping": ["prototyping", "prototype"],
    "Design systems": ["design systems", "design system"],

    # ── Soft / process ────────────────────────────────────────────────────────
    "Agile methodology": ["agile", "scrum", "kanban", "sprint"],
    "System design": ["system design", "distributed systems", "scalable architecture"],
    "Code review": ["code review", "peer review"],
    "REST API design": ["rest api", "restful api", "rest apis", "restful"],
    "GraphQL": ["graphql"],
    "gRPC": ["grpc"],
    "Microservices": ["microservices", "microservice architecture", "service mesh"],
    "Event-driven architecture": ["event-driven", "event sourcing", "message queue", "rabbitmq"],
    "Object-oriented programming": ["oop", "object-oriented", "object oriented programming"],
    "Data structures and algorithms": ["data structures", "algorithms", "dsa"],

    # ── BI / BA ──────────────────────────────────────────────────────────────
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Requirement gathering": ["requirement gathering", "requirements analysis", "brd", "frs"],
    "Stakeholder management": ["stakeholder management", "stakeholder communication"],
    "Process improvement": ["process improvement", "bpmn", "business process"],

    # ── Web fundamentals (previously missing) ────────────────────────────────
    "HTML": ["html", "html5", "hypertext markup language"],
    "CSS": ["css", "css3", "cascading style sheets", "sass", "scss", "styled components"],
    "Bootstrap": ["bootstrap", "bootstrap 5", "bootstrap4"],
    "Tailwind CSS": ["tailwind", "tailwind css"],
    "Webpack": ["webpack", "vite", "rollup", "parcel"],

    # ── ML / AI tools (previously missing) ───────────────────────────────────
    "Convolutional Neural Network": ["cnn", "convolutional neural network", "convnet"],
    "OpenCV": ["opencv", "open cv", "cv2", "computer vision library"],
    "spaCy": ["spacy", "spacy nlp", "en_core_web_sm", "en core web"],
    "Streamlit": ["streamlit"],
    "Sentence-BERT": ["sbert", "sentence-bert", "sentence bert",
                      "sentencebert", "sentence transformers", "sentence_transformers"],
    "Logistic Regression": ["logistic regression", "logistic reg", "lr classifier"],
    "TF-IDF": ["tf-idf", "tfidf", "tf idf", "term frequency inverse document frequency"],
    "Cosine Similarity": ["cosine similarity", "cosine sim", "cosine distance"],
    "Hugging Face": ["hugging face", "huggingface", "transformers library"],
    "LangChain": ["langchain", "lang chain"],
    "Weights & Biases": ["weights and biases", "wandb", "weights & biases"],
    "FastAI": ["fastai", "fast.ai"],
    "YOLO": ["yolo", "yolov5", "yolov8", "object detection yolo"],
    "Stable Diffusion": ["stable diffusion", "diffusion model", "diffusers"],

    # ── Core CS concepts (previously missing) ─────────────────────────────────
    "Computer networks": ["computer networks", "computer networking", "networking",
                          "tcp/ip", "http", "dns", "osi model"],
    "Database management systems": ["dbms", "database management", "database management systems",
                                    "relational database", "rdbms"],
    "Operating systems": ["operating systems", "operating system concepts", "os concepts",
                          "linux administration", "unix"],
    "Compiler design": ["compiler design", "compilers", "lexical analysis"],
    "Computer architecture": ["computer architecture", "computer organisation", "coa"],

    # ── Additional languages (previously missing) ─────────────────────────────
    "C (programming language)": ["c programming", " c ", "c language", "c99", "c11"],
    "C#": ["c#", "csharp", "c sharp", ".net", "dotnet", "asp.net"],
    "Rust": ["rust", "rust lang", "rust programming"],
    "Scala": ["scala"],
    "MATLAB": ["matlab"],
    "Julia": ["julia", "julia lang"],
    "Dart": ["dart", "dart programming"],
    "Flutter": ["flutter"],

    # ── Additional frameworks & tools (previously missing) ────────────────────
    "FastAPI": ["fastapi", "fast api"],   # duplicate-safe, already present but adding alias
    "Celery": ["celery", "celery worker", "async task queue"],
    "Socket.io": ["socket.io", "socketio", "websockets", "websocket"],
    "Nginx": ["nginx", "reverse proxy"],
    "PM2": ["pm2", "process manager"],
    "Gunicorn": ["gunicorn", "uwsgi"],
    "Swagger": ["swagger", "openapi", "api documentation"],
    "Postman": ["postman"],
    "VS Code": ["vs code", "vscode", "visual studio code"],
    "Jupyter": ["jupyter", "jupyter notebook", "ipynb", "google colab", "colab"],
    "Apache Hadoop": ["hadoop", "hdfs", "mapreduce", "hive", "pig"],
    "Databricks": ["databricks", "delta lake"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "microsoft excel", "spreadsheet", "pivot table"],
    "Power Automate": ["power automate", "power apps", "microsoft power platform"],
    "SAP": ["sap", "sap erp", "sap hana"],
    "Salesforce": ["salesforce", "crm", "salesforce crm"],
    "Jira": ["jira", "atlassian jira", "project tracking"],
    "Confluence": ["confluence", "atlassian confluence"],
    "Figma": ["figma"],   # already present, safe duplicate
    "Canva": ["canva"],
    "Arduino": ["arduino", "raspberry pi", "iot", "embedded systems"],

    # ── Soft skills & methodologies ───────────────────────────────────────────
    "Communication skills": ["communication", "verbal communication",
                              "written communication", "presentation skills"],
    "Problem solving": ["problem solving", "problem-solving", "analytical skills",
                        "critical thinking"],
    "Team leadership": ["team leadership", "team lead", "leadership", "team management"],
    "Time management": ["time management", "deadline management"],
    "Project management": ["project management", "project lead", "project coordinator",
                           "pmp", "prince2"],
}

# Build a fast lookup: normalised surface → canonical name
_LOOKUP: dict[str, str] = {}
for canonical, aliases in ESCO_SKILL_MAP.items():
    for alias in aliases:
        _LOOKUP[alias.strip().lower()] = canonical
# Also add canonical itself
for canonical in ESCO_SKILL_MAP:
    _LOOKUP[canonical.lower()] = canonical


def _normalise(text: str) -> str:
    """Lowercase and strip extra whitespace."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def extract_esco_skills(text: str) -> List[str]:
    """
    Extract ESCO-canonical skill names from resume or job description text.

    Strategy:
      1. Phrase-based scan (longest match first) for multi-word skills
      2. Token-based scan for single-word skills
      3. Fallback: colon-split section lines (catches unlisted skills)

    Args:
        text: raw resume or job description string

    Returns:
        Deduplicated list of canonical ESCO skill names, sorted alphabetically.
    """
    found: Set[str] = set()
    normalised = _normalise(text)

    # Pass 1: multi-word phrase scan (sort by length desc for longest-match)
    phrases_sorted = sorted(_LOOKUP.keys(), key=len, reverse=True)
    working = normalised

    for phrase in phrases_sorted:
        if len(phrase.split()) < 2:
            continue
        # Use word-boundary-like match
        pattern = r'(?<!\w)' + re.escape(phrase) + r'(?!\w)'
        if re.search(pattern, working):
            found.add(_LOOKUP[phrase])
            # Mask matched phrase so it doesn't double-match substrings
            working = re.sub(pattern, ' __matched__ ', working)

    # Pass 2: single-word token scan
    tokens = re.findall(r'\b\w[\w\.\/\+\#]*\b', working)
    for token in tokens:
        t = token.lower()
        if t in _LOOKUP:
            found.add(_LOOKUP[t])

    # Pass 3: colon-split fallback (for skills listed as "Languages: Python, Java")
    for line in text.split('\n'):
        if ':' in line:
            skill_part = line.split(':', 1)[1]
            for item in re.split(r'[,;/|]', skill_part):
                item_clean = _normalise(item)
                if item_clean in _LOOKUP:
                    found.add(_LOOKUP[item_clean])
                # Also try each word
                for word in item_clean.split():
                    if word in _LOOKUP:
                        found.add(_LOOKUP[word])

    return sorted(found)


def esco_overlap_score(resume_skills: List[str],
                       jd_skills: List[str]) -> float:
    """
    Compute Jaccard-based skill overlap between a resume and a job description.

    Score = |intersection| / |union|

    Returns float in [0, 1].
    Returns 0.0 if either list is empty.
    """
    if not resume_skills or not jd_skills:
        return 0.0

    set_r = set(resume_skills)
    set_j = set(jd_skills)

    intersection = set_r & set_j
    union = set_r | set_j

    return len(intersection) / len(union)


def esco_overlap_score_soft(resume_skills: List[str],
                             jd_skills: List[str]) -> float:
    """
    Soft overlap: |intersection| / |jd_skills|
    (what fraction of JD-required skills does the resume cover?)
    Returns float in [0, 1].
    """
    if not jd_skills:
        return 0.0
    intersection = set(resume_skills) & set(jd_skills)
    return len(intersection) / len(jd_skills)


def combined_score(sbert_score: float,
                   resume_text: str,
                   jd_text: str,
                   sbert_weight: float = 0.65,
                   esco_weight: float = 0.35) -> Tuple[float, List[str], List[str], float]:
    """
    Combine SBERT cosine similarity with ESCO skill overlap score.

    Args:
        sbert_score  : cosine similarity from SBERT (0-1)
        resume_text  : raw resume text
        jd_text      : raw job description text
        sbert_weight : weight for SBERT score (default 0.65)
        esco_weight  : weight for ESCO overlap (default 0.35)

    Returns:
        (final_score, resume_skills, jd_skills, esco_score)
    """
    resume_skills = extract_esco_skills(resume_text)
    jd_skills     = extract_esco_skills(jd_text)
    esco_score    = esco_overlap_score_soft(resume_skills, jd_skills)
    final         = (sbert_score * sbert_weight) + (esco_score * esco_weight)
    return round(final, 4), resume_skills, jd_skills, round(esco_score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    resume = """
    Senior Python Developer with 5 years building Django REST APIs and FastAPI services.
    Technical Skills: Python, Django, Flask, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, AWS.
    Experienced with CI/CD pipelines using GitHub Actions and Jenkins.
    Certification: AWS Certified Developer Associate.
    Education: B.Tech Computer Science from IIT Bombay (2019).
    Key Project: Built a real-time data pipeline using Kafka and Apache Spark processing 1M events/day.
    """

    jd = """
    We are looking for a Backend Engineer with:
    - Strong Python (Django or FastAPI preferred)
    - Experience with PostgreSQL and Redis
    - Docker and Kubernetes for deployments
    - REST API design experience
    - AWS (Lambda, EC2, S3)
    - CI/CD familiarity (GitHub Actions, Jenkins)
    - Kafka or RabbitMQ is a plus
    """

    r_skills = extract_esco_skills(resume)
    j_skills  = extract_esco_skills(jd)

    print("Resume ESCO skills:")
    for s in r_skills:
        print(f"  • {s}")

    print("\nJD ESCO skills:")
    for s in j_skills:
        print(f"  • {s}")

    jaccard = esco_overlap_score(r_skills, j_skills)
    soft    = esco_overlap_score_soft(r_skills, j_skills)
    print(f"\nJaccard overlap:  {jaccard:.4f}")
    print(f"Soft overlap:     {soft:.4f}  (fraction of JD skills covered)")

    # Simulate combined score with mock SBERT
    mock_sbert = 0.72
    final, _, _, esco_s = combined_score(mock_sbert, resume, jd)
    print(f"\nMock SBERT score: {mock_sbert}")
    print(f"ESCO score:       {esco_s}")
    print(f"Combined score:   {final}  (SBERT×0.65 + ESCO×0.35)")