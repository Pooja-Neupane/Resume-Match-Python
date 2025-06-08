from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Resume:
    def __init__(self, content):
        self.content = content

class Job:
    def __init__(self, title, description):
        self.title = title
        self.description = description

class ResumeMatcher:
    def __init__(self, resume, jobs):
        self.resume = resume
        self.jobs = jobs

    def match_jobs(self):
        texts = [self.resume.content] + [job.description for job in self.jobs]
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(texts)

        # Calculate similarity of resume with each job description
        similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        
        job_matches = []
        for i, score in enumerate(similarities):
            job_matches.append((self.jobs[i].title, round(score * 100, 2)))

        job_matches.sort(key=lambda x: x[1], reverse=True)
        return job_matches


# ========== Sample Execution ==========

# User's resume content
my_resume = Resume("Python developer with experience in machine learning, data analysis, and web development.")

# Job listings
jobs = [
    Job("Web Developer", "Looking for someone with experience in HTML, CSS, JavaScript, and web development."),
    Job("Data Scientist", "Requires knowledge in machine learning, Python, and data analytics."),
    Job("Mobile App Developer", "Seeking Android developer with skills in Java and Kotlin."),
    Job("AI Engineer", "Must have experience with machine learning, deep learning, and Python.")
]

# Matching
matcher = ResumeMatcher(my_resume, jobs)
results = matcher.match_jobs()

# Display matched jobs
print("🔍 Recommended Jobs (by Relevance %):")
for title, score in results:
    print(f"{title}: {score}% match")
