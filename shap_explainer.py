

import numpy as np
import joblib
import re


def explain_prediction_fast(pipeline, cleaned_text: str, predicted_class: str,
                             top_n: int = 15) -> dict:
    """
    Returns top positive and negative feature contributions for the predicted class.

    Args:
        pipeline       : loaded sklearn Pipeline (TF-IDF + LR)
        cleaned_text   : pre-cleaned resume text (lowercase, no punctuation)
        predicted_class: predicted category string
        top_n          : how many top features to return each side

    Returns:
        {"class": str, "positive": [(word, val),...], "negative": [(word, val),...]}
    """
    tfidf   = pipeline.named_steps['tfidf']
    clf     = pipeline.named_steps['clf']
    classes = list(clf.classes_)

    if predicted_class not in classes:
        return {"class": predicted_class, "positive": [], "negative": []}

    class_idx = classes.index(predicted_class)

# Transform text to TF-IDF vector
    X_vec    = tfidf.transform([cleaned_text])
    X_dense  = X_vec.toarray()[0]           # shape (n_features,)

# Contribution = coefficient × TF-IDF weight  (this IS what SHAP LinearExplainer computes
    coef          = clf.coef_[class_idx]    # shape (n_features,)
    contributions = coef * X_dense          # element-wise

    feature_names = np.array(tfidf.get_feature_names_out())

# Only show features actually present in this resume (non-zero TF-IDF
    nonzero_mask    = X_dense > 0
    active_contribs = contributions[nonzero_mask]
    active_names    = feature_names[nonzero_mask]

    if len(active_contribs) == 0:
        return {"class": predicted_class, "positive": [], "negative": []}

    sorted_idx = np.argsort(active_contribs)[::-1]

    positive = [
        (str(active_names[i]), float(active_contribs[i]))
        for i in sorted_idx[:top_n]
        if active_contribs[i] > 0
    ]
    negative = [
        (str(active_names[i]), float(active_contribs[i]))
        for i in sorted_idx[-top_n:]
        if active_contribs[i] < 0
    ]

    return {
        "class":    predicted_class,
        "positive": positive,
        "negative": negative,
    }


def render_shap_html(explanation: dict) -> str:
    """
    Renders explanation as HTML for st.components.v1.html().
    Styled for Streamlit dark theme.
    """
    pos = explanation["positive"]
    neg = explanation["negative"]
    cls = explanation["class"]

    if not pos and not neg:
        return (
            f'<div style="font-family:sans-serif;padding:12px;color:#aaa;">'
            f'No significant features found for <em>{cls}</em>. '
            f'The resume may contain mostly words not in the training vocabulary.</div>'
        )

    max_val = max([v for _, v in pos] + [abs(v) for _, v in neg] + [0.001])

    def bar(word, val):
        pct   = min(abs(val) / max_val * 100, 100)
        color = "#1E88E5" if val > 0 else "#E53935"
        bg    = "#0d1117"
        return (
            f'<div style="display:flex;align-items:center;margin:5px 0;'
            f'background:{bg};border-radius:4px;padding:2px 4px;">'
            f'<span style="width:210px;font-size:13px;color:#e0e0e0;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis;flex-shrink:0;">{word}</span>'
            f'<div style="background:{color};height:14px;width:{pct:.1f}%;'
            f'border-radius:3px;margin-left:8px;min-width:4px;"></div>'
            f'<span style="margin-left:10px;font-size:12px;color:{color};'
            f'white-space:nowrap;">{val:+.4f}</span>'
            f'</div>'
        )

    pos_html = "".join(bar(w, v) for w, v in pos)
    neg_html = "".join(bar(w, v) for w, v in neg) if neg else (
        '<span style="color:#888;font-size:13px;">None detected</span>'
    )

    html = f"""
    <div style="font-family:'Segoe UI',sans-serif;padding:14px;
                background:#161b22;color:#c9d1d9;border-radius:8px;
                border:1px solid #30363d;">
      <h4 style="margin:0 0 4px;color:#58A6FF;font-size:15px;">
        Why <em style="color:#79C0FF;">{cls}</em>?
        &nbsp;<span style="font-size:11px;font-weight:normal;color:#8B949E;">
        (TF-IDF weight × LR coefficient)</span>
      </h4>
      <p style="font-size:12px;color:#6E7681;margin:0 0 12px;">
        <span style="color:#388BFD;">■</span>&nbsp;Blue = words pushing
        <b style="color:#79C0FF;">toward</b> this category &nbsp;
        <span style="color:#F85149;">■</span>&nbsp;Red = words pushing
        <b style="color:#F85149;">away</b> from it.
      </p>

      <div style="font-size:13px;font-weight:600;color:#3FB950;margin-bottom:6px;">
        ▲ Top {len(pos)} positive features
      </div>
      <div style="margin-bottom:16px;">{pos_html}</div>

      <div style="font-size:13px;font-weight:600;color:#F85149;margin-bottom:6px;">
        ▼ Top {len(neg)} negative features
      </div>
      <div>{neg_html}</div>
    </div>
    """
    return html


# Standalone test
if __name__ == "__main__":
    import sys, os

    pipeline_path = (
        sys.argv[1] if len(sys.argv) > 1
        else "model/final_resume_classifier_logistic.pkl"
    )
    if not os.path.exists(pipeline_path):
        print(f"❌ Model not found at '{pipeline_path}'")
        print("   Run: python train_tfidf_lr.py first.")
        sys.exit(1)

    print(f"Loading model from {pipeline_path} ...")
    pipe = joblib.load(pipeline_path)

# Simulate Abhinav's resume
    sample = re.sub(r'[^\w\s]', '', """
    java c python javascript html css bootstrap react nodejs expressjs flask
    streamlit tensorflow spacy opencv scikit learn pandas numpy cnn nlp tfidf
    logistic regression sentencebert mysql postgresql mongodb sqlite oop
    operating systems dbms computer networks machine learning emotion detection
    waste management resume filtering ats classifier
    """.lower())

    pred  = pipe.predict([sample])[0]
    proba = pipe.predict_proba([sample])[0]
    top5  = sorted(zip(pipe.classes_, proba), key=lambda x: -x[1])[:5]

    print(f"\nPredicted: {pred}")
    print("Top 5:")
    for cat, p in top5:
        print(f"  {cat:35s} {p*100:.2f}%")

    expl = explain_prediction_fast(pipe, sample, pred)
    print(f"\nContributions for '{pred}':")
    print("▲ Positive:")
    for w, v in expl["positive"][:10]:
        print(f"    {w:30s} {v:+.4f}")
    print("▼ Negative:")
    for w, v in expl["negative"][:5]:
        print(f"    {w:30s} {v:+.4f}")

    html = render_shap_html(expl)
    out = "shap_test_output.html"
    with open(out, "w") as f:
        f.write(f"<html><body style='background:#0d1117;padding:20px'>{html}</body></html>")
    print(f"\n✅ HTML saved to {out}")