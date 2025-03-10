#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

#%%
df = pd.read_csv("./Cache/brunch_data.csv")
ids = df.business_id.to_list()

#%%

def get_rows(filepath, key, values):
    matches = []
    values = set(values)  # Convert to set for faster lookup
    
    with open(filepath) as f:
        for line in f:
            row = json.loads(line)
            if row[key] in values:
                matches.append(row) 
                
    return pd.DataFrame(matches)

df = get_rows(
    './Data/yelp_dataset/yelp_academic_dataset_review.json',
    key='business_id',
    values=set(ids)  # Convert to set for faster lookup
)


# %%

df.to_csv("./Cache/brunch_reviews.csv", index=False)
# %%

#filter out non-english reviews
def fast_detect_english(text):
    text = str(text)[:100].lower()
    
    english_patterns = r'\b(the|be|to|of|and|a|in|that|have|i|it|for|not|on|with|he|as|you|do|at)\b'
    
    english_count = len(re.findall(english_patterns, text.lower()))
    
    # Check if text uses mainly ASCII characters
    ascii_ratio = len([c for c in text if ord(c) < 128]) / (len(text) or 1)
    
    return english_count >= 2 and ascii_ratio > 0.8

# df['is_english'] = df['text'].apply(fast_detect_english)