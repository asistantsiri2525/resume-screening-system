from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

app = Flask(__name__)

# 🔹 PDF text extract
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    top_candidate = None
    total_resumes = 0
    average_score = 0

    if request.method == "POST":
        jd = request.form.get("jd")
        files = request.files.getlist("cv_files")

        if jd and files:
            for file in files:
                if file.filename != "":
                    cv_text = extract_text_from_pdf(file)

                    # 🔹 Similarity
                    texts = [jd, cv_text]
                    vectorizer = TfidfVectorizer()
                    vectors = vectorizer.fit_transform(texts)

                    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                    similarity_score = similarity * 100

                    # 🔹 Skills
                    skills_list = [
                        "python", "sql", "excel", "machine learning",
                        "data analysis", "pandas", "statistics",
                        "data visualization", "power bi"
                    ]

                    clean_jd = jd.lower()
                    clean_cv = cv_text.lower()

                    matched = [skill for skill in skills_list if skill in clean_cv]
                    missing = [skill for skill in skills_list if skill in clean_jd and skill not in clean_cv]

                    # 🔹 Skill score
                    total_skills = len([skill for skill in skills_list if skill in clean_jd])
                    matched_count = len([skill for skill in skills_list if skill in clean_cv and skill in clean_jd])

                    if total_skills > 0:
                        skill_score = (matched_count / total_skills) * 100
                    else:
                        skill_score = 0

                    # 🔥 Final Score (with bonus)
                    score = (0.4 * similarity_score) + (0.6 * skill_score)

                    if skill_score >= 70:
                        score += 10

                    score = round(min(score, 100), 2)

                    # 🔹 Result
                    if score >= 70:
                        result_msg = "Excellent Match ✅"
                    elif score >= 40:
                        result_msg = "Good Match 👍"
                    else:
                        result_msg = "Low Match ❌"

                    # 🔹 Store
                    results.append({
                        "name": file.filename,
                        "score": score,
                        "result": result_msg,
                        "matched": ", ".join(matched),
                        "missing": ", ".join(missing)
                    })

            # 🔥 Sort
            results = sorted(results, key=lambda x: x["score"], reverse=True)

            # 🔥 Dashboard FIX
            total_resumes = len(results)

            if results:
                top_candidate = results[0]
                average_score = round(sum(r["score"] for r in results) / total_resumes, 2)

    return render_template("index.html",
                           results=results,
                           top_candidate=top_candidate,
                           total_resumes=total_resumes,
                           average_score=average_score)


if __name__ == "__main__":
    app.run(debug=True)