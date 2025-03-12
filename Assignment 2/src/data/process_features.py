#%%

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline
import ollama
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#%%

feature_importance = pd.read_csv("../../data/processed/tfidf/feature_importance.csv")
# %%

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast & lightweight


#Convert Words to Embeddings
embeddings = model.encode(feature_importance['term'].tolist(), convert_to_numpy=True)


# %%
#Cluster Words using KMeans
num_clusters = 25  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
feature_importance['cluster'] = kmeans.fit_predict(embeddings)


# %%
# Sorts by cluster → Ensures words within the same cluster stay together.
# Sorts by coefficient (descending) → Prioritizes the most important words first.
# This makes sense because when generating labels using GPT, we want the most important words for each cluster to be presented first. GPT will likely make better labels when it sees the top TF-IDF terms first.
df_sorted = feature_importance.sort_values(by=["cluster", "coefficient"], ascending=[True, False])


# %%
# Generate Cluster Labels using Ollama
# label_generator = pipeline("text-generation",
#                            model="mistralai/Mistral-7B-Instruct-v0.2",
#                            max_new_tokens=10)

def generate_label(terms):
    """Use a local LLM via Ollama to generate a cluster label."""
    prompt = f"Summarize the category for these words in 3 words or less: {terms}. Return only one phrase, no other text."
    response = ollama.chat(model="llama3.2:latest",
                           messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()


cluster_labels = {}
for cluster in feature_importance['cluster'].unique():
    terms = ", ".join(df_sorted[df_sorted['cluster'] == cluster]['term'].tolist())
    response = generate_label(terms)
    cluster_labels[cluster] = response

# Map Generated Labels to Clusters
feature_importance['cluster_label'] = feature_importance['cluster'].map(cluster_labels)

# %%
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                      c=feature_importance['cluster'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Cluster")
plt.title("TF-IDF Term Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# Annotate terms on the scatter plot
for i, txt in enumerate(feature_importance['term']):
    plt.annotate(txt, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

plt.show()


# %%
mean_coefficients = {}
for cluster in feature_importance['cluster'].unique():
    mean_coefficient = feature_importance[feature_importance['cluster'] == cluster]['coefficient'].mean()
    mean_coefficients[cluster] = mean_coefficient

feature_importance['mean_coefficient'] = feature_importance['cluster'].map(mean_coefficients)

# %%
feature_importance.groupby('cluster_label')['mean_coefficient'].mean().sort_values(ascending=False)

# %%
