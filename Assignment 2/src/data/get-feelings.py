#%%
import pandas as pd
import swifter
import numpy as np
import time
import logging
import os
import concurrent.futures
import tiktoken
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Change this import
import multiprocessing as mp
import ast



import matplotlib.pyplot as plt
import seaborn as sns

from transformers import pipeline
from transformers import BertTokenizer

from torch.utils.data import DataLoader
import torch
import gc



sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 5]

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# %%

# df = pd.read_csv(os.path.join(BASE_DIR, "data/interim/text-analysis/brunch_reviews_text_analysis.csv"))
df_feelings = pd.read_csv(os.path.join(BASE_DIR, "data/interim/feelings/feelings_results_final.csv"))

# %%
# emotion_pipeline = pipeline(
#     "text-classification",
#     model="SamLowe/roberta-base-go_emotions",
#     device='mps',  # Using your existing MPS setup
#     top_k=None
# )
# %%
# batch_size = 256

# def process_batch(batch):
#     with torch.inference_mode(): 
#         return emotion_pipeline(batch, truncation=True, max_length=256, batch_size=len(batch))

# def process_reviews(df, output_file="../../data/processed/feelings/feelings_results_final.csv"):
#     texts = df['text']  # Avoid converting to list (saves memory)
#     num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)  # Precompute batch count

#     first_batch = True  # To write header only once
    
#     with open(output_file, 'w', encoding='utf-8') as f:  # Open file once
#         for i in tqdm(range(num_batches), desc="Processing reviews", unit="batch"):
#             batch = texts[i * batch_size: (i + 1) * batch_size].tolist()  # Convert only the current batch to list
#             results = process_batch(batch)  # Process batch
#             batch_df = pd.DataFrame(results)

#             batch_df.to_csv(f, mode='a', index=False, header=first_batch, encoding='utf-8')  # Append to file
            
#             first_batch = False  # Ensure header is written only for the first batch

#             del results, batch_df  # Free memory
#             gc.collect()

# torch.mps.empty_cache()  # Clear GPU memory
# gc.collect()  # Clear CPU memory

# process_reviews(df)

# gc.collect()
# torch.mps.empty_cache()



# %%

# df_feelings['feelings'] = df_feelings.to_numpy().tolist()
# df_feelings = df_feelings[['feelings']]
# df_feelings.to_csv(os.path.join(BASE_DIR, "data/interim/feelings/feelings_results_final.csv"), index=False)

df_feelings['dummy'] = 1

#%%
def process_feelings(col):
    rows = []
    for row in tqdm(col, desc="Processing rows"):
        parsed_row = []
        try:
            # First, parse the outer list
            row_list = ast.literal_eval(row)
            
            # Then, parse each inner string (dictionary)
            for f in row_list:
                parsed_row.append(ast.literal_eval(f))
                
            rows.append(parsed_row)
        except Exception as e:
            print(f"Error parsing row: {row}\n{e}")

    rows_parsed = []
    
    for row in tqdm(rows, desc="Processing feelings"):
        dicts = {f['label']: f['score'] for f in row}
        dicts = dict(sorted(dicts.items()))
        dicts = {k: round(v, 4) for k, v in dicts.items()}
        
        rows_parsed.append(dicts)

    rows_parsed = pd.DataFrame(rows_parsed)
    
    return rows_parsed

# Example usage
procesesed_feelings = process_feelings(df_feelings.feelings)
procesesed_feelings.to_csv(os.path.join(BASE_DIR, "data/interim/feelings/feelings_results_parsed.csv"), index=False)




