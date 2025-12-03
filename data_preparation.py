import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Load data
data = pd.read_csv("Coursera2021.csv")
data = data[['Course Name', 'Difficulty Level', 'Course Description', 'Skills']]

# Preprocess data
data['Course Name'] = data['Course Name'].str.replace(' ', ',').str.replace(',,', ',').str.replace(':', '')
data['Course Description'] = data['Course Description'].str.replace(' ', ',').str.replace(',,', ',').str.replace('_', '').str.replace(':', '').str.replace('(', '').str.replace(')', '')
data['Skills'] = data['Skills'].str.replace('(', '').str.replace(')', '')

data['tags'] = data['Course Name'] + data['Difficulty Level'] + data['Course Description'] + data['Skills']
new_df = data[['Course Name', 'tags']]
new_df['tags'] = data['tags'].str.replace(',', ' ')
new_df['Course Name'] = data['Course Name'].str.replace(',', ' ')
new_df.rename(columns={'Course Name': 'course_name'}, inplace=True)
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Stemming
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

# Vectorization and similarity calculation
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# Save models and data
with open('cv_model.pkl', 'wb') as file:
    pickle.dump(cv, file)

with open('similarity_matrix.pkl', 'wb') as file:
    pickle.dump(similarity, file)

with open('courses_data.pkl', 'wb') as file:
    pickle.dump(new_df, file)
