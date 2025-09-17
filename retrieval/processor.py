from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextProcessor:
    def clean_text(self, text):
        text = str(text).lower()
        import re
        text = re.sub(r"[^a-zA-Z0-9\sáéíóúäëïöüñ]", "", text)
        return text

    def compute_similarity(self, claim, evidence_list):
        """
        Compute average cosine similarity between claim and each evidence text
        """
        texts = [claim] + evidence_list
        vectorizer = TfidfVectorizer().fit_transform(texts)
        cosine_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])
        # Return average similarity
        return cosine_matrix.mean()

