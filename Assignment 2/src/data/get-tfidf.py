#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from langdetect import detect
import re
import scipy.sparse as sparse  


from sklearn.feature_extraction.text import TfidfVectorizer


#%%

df = pd.read_csv("../../data/processed/brunch_reviews_english.csv")

#%%


vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['text'])

# Calculate mean scores directly on sparse matrix
top_words = pd.Series(
    np.asarray(tfidf_matrix.mean(axis=0)).ravel(),
    index=vectorizer.get_feature_names_out()
).sort_values(ascending=False)




# %%

sparse.save_npz("./Data/Generated/tfidf_matrix.npz", tfidf_matrix)

#%%
# Save metadata (business IDs, feature names, and their mapping)
metadata = {
    'business_ids': df['business_id'].tolist(),
    'feature_names': vectorizer.get_feature_names_out().tolist(),
    'shape': tfidf_matrix.shape
}
with open('../../data/processed/tfidf_metadata.json', 'w') as f:
    json.dump(metadata, f)


# %%
# Print top 10 terms and their average TF-IDF scores
print("\nTop 10 terms by average TF-IDF score:")
print(top_words.head(10))

# Show example of one review's TF-IDF values
print("\nExample review TF-IDF values (non-zero terms):")
example_review = pd.Series(
    tfidf_matrix[0].toarray()[0],
    index=vectorizer.get_feature_names_out()
)
print(example_review[example_review > 0].sort_values(ascending=False).head())
# %%
