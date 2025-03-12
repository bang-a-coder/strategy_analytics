#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from langdetect import detect
import re
import scipy.sparse as sparse  

from sklearn.metrics import classification_report, accuracy_score
import statsmodels.api as sm
import seaborn as sns


from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#%%

# df = pd.read_csv("../../data/processed/brunch_reviews_english.csv")

#%%


# vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
# tfidf_matrix = vectorizer.fit_transform(df['text'])

# # Calculate mean scores directly on sparse matrix
# top_words = pd.Series(
#     np.asarray(tfidf_matrix.mean(axis=0)).ravel(),
#     index=vectorizer.get_feature_names_out()
# ).sort_values(ascending=False)




# %%

# sparse.save_npz("./Data/Generated/tfidf_matrix.npz", tfidf_matrix)

# #%%
# # Save metadata (business IDs, feature names, and their mapping)
# metadata = {
#     'business_ids': df['business_id'].tolist(),
#     'feature_names': vectorizer.get_feature_names_out().tolist(),
#     'shape': tfidf_matrix.shape
# }
# with open('../../data/processed/tfidf_metadata.json', 'w') as f:
#     json.dump(metadata, f)


# # %%
# # Print top 10 terms and their average TF-IDF scores
# print("\nTop 10 terms by average TF-IDF score:")
# print(top_words.head(10))

# # Show example of one review's TF-IDF values
# print("\nExample review TF-IDF values (non-zero terms):")
# example_review = pd.Series(
#     tfidf_matrix[0].toarray()[0],
#     index=vectorizer.get_feature_names_out()
# )
# print(example_review[example_review > 0].sort_values(ascending=False).head())

# %%
# df_tfidf = pd.read_csv("../../data/processed/tfidf/tfidf_matrix.npz")
# %%

# Load the TF-IDF matrix and metadata
tfidf_matrix = sparse.load_npz("../../data/processed/tfidf/tfidf_matrix.npz")
with open('../../data/processed/tfidf/tfidf_metadata.json', 'r') as f:
    metadata = json.load(f)

# Load the reviews data for ratings
df = pd.read_csv("../../data/processed/brunch_reviews_text_analysis_lite.csv")


#%%

df = df[['review_id', 'stars_binary']]

df.stars_binary = df.stars_binary.apply(lambda x: 1 if x == 'High' else 0)

#%%

# Split the data
X = tfidf_matrix
y = df['stars_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")

# Analyze feature importance
feature_importance = pd.DataFrame({
    'term': metadata['feature_names'],
    'coefficient': model.coef_
}).sort_values('coefficient', ascending=False)

# %%
feature_importance.head(10)
# %%

from sklearn.metrics import confusion_matrix

# Binarize predictions
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

# Compute confusion matrix
cm = confusion_matrix( y_pred_binary, y_test)

print("\nClassification Report:")
print(classification_report(y_pred_binary, y_test))
print(f"Accuracy: {accuracy_score(y_pred_binary, y_test):.3f}")

# Pretty confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Predicted Label')
plt.xlabel('True Label')
plt.show()



# %%
import tensorflow as tf  # Uses Metal Performance Shaders (MPS) backend on M2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Assuming tfidf_matrix is a SciPy sparse matrix (e.g., from TfidfVectorizer)
# Convert to TensorFlow SparseTensor
X_tf = tf.sparse.SparseTensor(
    indices=np.array(tfidf_matrix.nonzero()).T,  # Get nonzero indices
    values=tfidf_matrix.data,  # Get values
    dense_shape=tfidf_matrix.shape  # Set shape
)

# Reorder SparseTensor to ensure indices are sorted
X_tf = tf.sparse.reorder(X_tf)

# Convert SparseTensor to Dense
X_tf_dense = tf.sparse.to_dense(X_tf)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tf_dense.numpy(), y, test_size=0.2, random_state=42)

# Create model
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))


# %%
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

epochs = range(1, len(history.history['loss']) + 1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.show()
# %%
