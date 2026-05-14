# STEP 1: Import Required Libraries

import nltk
import pandas as pd

# Download Required NLTK Packages

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# STEP 2: Sample Document

document = """
Natural Language Processing is a branch of Artificial Intelligence.
It helps computers understand human language.
Text preprocessing is important in NLP.
"""

print("ORIGINAL DOCUMENT")
print(document)

# STEP 3: Tokenization

from nltk.tokenize import word_tokenize

tokens = word_tokenize(document)

print("\nTOKENS")
print(tokens)

# STEP 4: POS Tagging

from nltk import pos_tag

pos = pos_tag(tokens)

print("\nPOS TAGGING")
print(pos)

# STEP 5: Stop Words Removal

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

filtered_words = []

for word in tokens:
    if word.lower() not in stop_words:
        filtered_words.append(word)

print("\nAFTER STOP WORD REMOVAL")
print(filtered_words)

# STEP 6: Stemming

from nltk.stem import PorterStemmer

ps = PorterStemmer()

stemmed_words = []

for word in filtered_words:
    stemmed_words.append(ps.stem(word))

print("\nSTEMMED WORDS")
print(stemmed_words)

# STEP 7: Lemmatization

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lemmatized_words = []

for word in filtered_words:
    lemmatized_words.append(lemmatizer.lemmatize(word))

print("\nLEMMATIZED WORDS")
print(lemmatized_words)

# STEP 8: TF-IDF Representation

from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Natural Language Processing is important",
    "NLP helps computers understand language",
    "Text preprocessing improves NLP"
]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

# Convert into DataFrame

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("\nTF-IDF REPRESENTATION")
print(tfidf_df)
# STEP 9: Plot Top Important Words using TF-IDF

import matplotlib.pyplot as plt
import numpy as np

# Calculate average TF-IDF score for each word

tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)

# Get feature names (words)

words = vectorizer.get_feature_names_out()

# Create DataFrame

tfidf_scores_df = pd.DataFrame({
    'Word': words,
    'TF-IDF Score': tfidf_scores
})

# Sort values in descending order

tfidf_scores_df = tfidf_scores_df.sort_values(
    by='TF-IDF Score',
    ascending=False
)

# Select Top 20 Words

top_words = tfidf_scores_df.head(20)

# Plot Graph

plt.figure(figsize=(12, 6))

plt.bar(
    top_words['Word'],
    top_words['TF-IDF Score'],
    color='skyblue'
)

plt.title('Top 20 Important Words by TF-IDF Score')

plt.xlabel('Words')

plt.ylabel('Average TF-IDF Score')

plt.xticks(rotation=45)

plt.show()