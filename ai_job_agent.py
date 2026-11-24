
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# ----------------------------------
# 1. TRAIN SMALL NEURAL NETWORK
# ----------------------------------

titles = [
    "AI Engineer",
    "Senior AI Engineer",
    "Artificial Intelligence Engineer",
    "Machine Learning Engineer",
    "Generative AI Engineer",
    "Deep Learning Engineer",
    "AI ML Developer",
    "NLP Engineer",
    "LLM Engineer",
    "AI Software Engineer",
    "Full Stack AI Engineer",
    "Applied AI Engineer",

    "Java Developer",
    "Accountant",
    "HR Manager",
    "Sales Executive",
    "Civil Engineer",
    "Graphic Designer",
    "Office Assistant",
    "Marketing Manager",
    "Network Administrator",
    "Finance Analyst",
    "Mechanical Engineer",
    "Customer Support Executive"
]

labels = [1] * 12 + [0] * 12  #1 means ai job and o means other jobs

#### TF-IDF Vectorizer -  Converts text into numerical features ####
#### Neural Network - Learns patterns from the numerical features  ####
######## 2 characters ==>	AI, I ,  E, En  3 characters ==> AI , I E,  En, Eng, 4 ==>  characters	AI E, I En,  Eng, Engi
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4)                 ###This means it extracts sequences containing 2, 3, or 4 characters.
    )),    
    ("neural_network", MLPClassifier(      ###Multi-Layer Perceptron Classifier.
        hidden_layer_sizes=(16, 8),
        max_iter=1000,                     ###It does not mean the model will always train exactly 1,000 times. Training may stop earlier when its convergence criteria are met.
        random_state=42                    ###This makes the model's random initialization and related random operations reproducible.
    ))
])

model.fit(titles, labels)


# ----------------------------------
# 2. TOOL: FETCH JOBS FROM API
# ----------------------------------

def fetch_jobs():
    jobs = []

    for page in range(1, 4):
        response = requests.get(
            "https://www.arbeitnow.com/api/job-board-api",
            params={"page": page},
            timeout=20
        )
        response.raise_for_status()
        jobs.extend(response.json().get("data", []))

    return jobs


# ----------------------------------
# 3. TOOL: CHECK TODAY'S DATE
# ----------------------------------

def posted_today(job):
    timestamp = job.get("created_at")

    if not timestamp:
        return False

    posted = datetime.fromtimestamp(
        int(timestamp),
        tz=timezone.utc
    ).astimezone(ZoneInfo("America/Detroit"))

    today = datetime.now(
        ZoneInfo("America/Detroit")
    ).date()

    return posted.date() == today


# ----------------------------------
# 4. AI AGENT WORKFLOW
# ----------------------------------

def run_agent():
    print("Searching today's AI Engineer jobs...\n")

    jobs = fetch_jobs()
    matches = []

    for job in jobs:
        title = job.get("title", "")

        if not posted_today(job):
            continue

        prediction = model.predict([title])[0]

        if prediction == 1:
            matches.append(job)

    print(f"Found {len(matches)} matching jobs\n")

    for job in matches:
        print("Job:", job.get("title"))
        print("Company:", job.get("company_name"))
        print("Location:", job.get("location"))
        print("Apply:", job.get("url"))
        print("-" * 50)

    if not matches:
        print("No matching jobs posted today were found.")


if __name__ == "__main__":
    run_agent()
