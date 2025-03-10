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

model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Much faster than RoBERTa
sentiment_pipeline = pipeline("sentiment-analysis",
                              model=model_name,
                              device='mps')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', device='mps')

# %%


def fast_count_tokens(texts, batch_size=2000):
    # GPT-2 encoding is a good default choice
    enc = tiktoken.get_encoding("gpt2")
    token_counts = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="Counting tokens"):
        batch = texts[i:i + batch_size]
        counts = [len(enc.encode(text)) for text in batch]
        token_counts.extend(counts)
    return token_counts

# token_counts = fast_count_tokens(df['text'].tolist())

# df['token_count'] = token_counts


# df.to_csv("../../data/interim/text-analysis/brunch_reviews_text_analysis.csv", index=False)

# %%

batch_size = 256

def process_batch(batch):
    with torch.inference_mode(): 
        return sentiment_pipeline(batch, truncation=True, max_length=256, batch_size=len(batch))

def process_reviews(df):
    texts = df['text'].tolist()
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    results = []
    for batch in tqdm(batches, desc="Processing reviews"):
        results.extend(process_batch(batch))
    
    final_df = pd.DataFrame(results, index=df.index)
    final_df.to_csv("../../data/interim/text-analysis/sentiment_results_final.csv")
    return final_df

# Get results
# torch.mps.empty_cache()  # Clear GPU memory
# sentiment_results = process_reviews(df)

# %%
sentiment_results = pd.read_csv("../../data/interim/text-analysis/sentiment_results_final.csv")
# %%

sentiment_results.drop(columns=['Unnamed: 0'], inplace=True)
sentiment_results.rename(columns={'label': 'sentiment', 'score': 'sentiment_score'}, inplace=True)

df = df.join(sentiment_results[['sentiment', 'sentiment_score']])



# %%
# df.to_csv("../../data/interim/text-analysis/brunch_reviews_text_analysis_with_sentiment.csv", index=False)
# %%
