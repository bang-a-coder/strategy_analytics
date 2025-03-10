#%%
#%%
#%%
import pandas as pd
import swifter
import numpy as np
import time
import logging
import os
import concurrent.futures
import tiktoken

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Change this import
import multiprocessing as mp




import matplotlib.pyplot as plt
import seaborn as sns

from transformers import pipeline
from transformers import BertTokenizer

from torch.utils.data import DataLoader
import torch



sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 5]

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Get absolute path
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# %%

df = pd.read_csv(os.path.join(BASE_DIR, "data/interim/text-analysis/brunch_reviews_text_analysis.csv"))

# %%
emotion_pipeline = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    device='mps',  # Using your existing MPS setup
    top_k=None
)
# %%
batch_size = 256

def process_batch(batch):
    with torch.inference_mode(): 
        return emotion_pipeline(batch, truncation=True, max_length=256, batch_size=len(batch))

def process_reviews(df):
    texts = df['text'].tolist()
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    results = []
    for batch in tqdm(batches, desc="Processing reviews"):
        results.extend(process_batch(batch))
    
    final_df = pd.DataFrame(results, index=df.index)
    final_df.to_csv("../../data/processed/feelings/feelings_results_final.csv")
    return final_df

# Get results
torch.mps.empty_cache()  # Clear GPU memory
sentiment_results = process_reviews(df.head(5000))
# %%
