

import os
import re
import numpy as np
import torch
from typing import List, Tuple


class DistilBertClassifier:
    """
    Thin wrapper around a saved DistilBertForSequenceClassification model.
    Thread-safe for Streamlit (load once, call predict repeatedly).
    """

    def __init__(self, model_dir: str = "distilbert_resume_model"):
        """
        Args:
            model_dir: path to the folder saved by train_distilbert_your_dataset.py
                       Must contain: config.json, pytorch_model.bin (or model.safetensors),
                       tokenizer files, label_classes.npy
        """
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"DistilBERT model folder not found: '{model_dir}'\n"
                "→ Run train_distilbert_your_dataset.py on Colab, download the zip, "
                "extract it, and place the folder next to app.py."
            )

        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        classes_path = os.path.join(model_dir, "label_classes.npy")
        self.classes = np.load(classes_path, allow_pickle=True).tolist()

        print(f"✅ DistilBERT loaded from '{model_dir}' on {self.device}")
        print(f"   Classes ({len(self.classes)}): {self.classes}")

    def _clean(self, text: str) -> str:
        return re.sub(r'[^\w\s]', ' ', text.lower()).strip()

    @torch.no_grad()
    def predict_proba(self, text: str) -> List[Tuple[str, float]]:
        """
        Returns list of (class, probability) sorted by probability descending.
        """
        inputs = self.tokenizer(
            self._clean(text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return sorted(zip(self.classes, probs.tolist()), key=lambda x: -x[1])

    def predict_top2(self, text: str) -> List[Tuple[str, float]]:
        """
        Returns top-2 (class, probability) tuples.
        Drop-in replacement for TF-IDF predict_proba top-2 block in app.py.
        """
        return self.predict_proba(text)[:2]

    def predict(self, text: str) -> str:
        """Returns the top predicted class name."""
        return self.predict_top2(text)[0][0]



if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "distilbert_resume_model"

    clf = DistilBertClassifier(model_dir)

    samples = [
        ("Python Developer",
         "Experienced Python Developer with 5 years at Zoho. "
         "Skills: Python, Django, FastAPI, PostgreSQL, Redis, Docker. "
         "Built inventory automation tool. B.Tech IIT Bombay."),
        ("DevOps Engineer",
         "DevOps Engineer specialising in CI/CD, Docker, Kubernetes, Terraform, AWS. "
         "Deployed production EKS clusters. CKA certified."),
        ("Data Scientist",
         "Data Scientist with 3 years at Fractal Analytics. "
         "Skills: Python, R, Machine Learning, Statistics, Tableau, SQL. "
         "Built customer churn model with 87% AUC."),
    ]

    print("\n── DistilBERT Predictions ──")
    for true_cat, text in samples:
        top2 = clf.predict_top2(text)
        pred = top2[0][0]
        match = "✅" if pred == true_cat else "❌"
        print(f"{match} True: {true_cat:30s} | Pred: {pred:30s} ({top2[0][1]*100:.1f}%)")
        print(f"   2nd: {top2[1][0]:30s} ({top2[1][1]*100:.1f}%)")
