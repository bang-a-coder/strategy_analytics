#%%
import pandas as pd
import swifter
import numpy as np
import time
import logging
import os
import concurrent.futures

from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm
import multiprocessing as mp




import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm



from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler


from transformers import pipeline
from transformers import BertTokenizer



sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 5]
 

#%%

df = pd.read_csv("./yelpsampple-updated.csv")

#%%

# Initialize emotion classifier
emotion_pipeline = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    device='mps',  # Using your existing MPS setup
    top_k=None
)

def process_emotions(texts, cache_dir="./emotion_cache", batch_size=1000):
    os.makedirs(cache_dir, exist_ok=True)
    all_results = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_file = f"{cache_dir}/batch_{i}.pkl"
        
        if os.path.exists(batch_file):
            all_results.extend(pd.read_pickle(batch_file))
            continue
            
        batch_results = emotion_pipeline(
            list(texts[i:i + batch_size]),
            truncation=True,
            padding=True,
            max_length=512,
            batch_size=32
        )
        pd.to_pickle(batch_results, batch_file)
        all_results.extend(batch_results)
    
    return all_results

emotions = process_emotions(df['review_text'])

#%%


