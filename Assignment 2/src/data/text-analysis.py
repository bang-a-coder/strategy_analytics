#%%
import pandas as pd

#%%
df = pd.read_csv("../../data/interim/text-analysis/brunch_reviews_text_analysis_with_sentiment.csv")

#%%

df.head()
# %%

df['word_count'] = df['text'].apply(lambda x: len(str(x).split(" ")))
df['char_count'] = df['text'].apply(lambda x: len(str(x)))
df['exclamation_count'] = df['text'].str.count('!')
df['question_mark_count'] = df['text'].str.count('\?')

# Count uppercase words (could indicate emphasis)
df['uppercase_word_count'] = df['text'].apply(lambda x: sum(1 for word in str(x).split() if word.isupper()))

# Count numeric digits
df['number_count'] = df['text'].str.count(r'\d')

# %%

# Add sentence count
df['sentence_count'] = df['text'].str.count(r'[.!?]+')

# Count common punctuation
df['comma_count'] = df['text'].str.count(',')
df['quotes_count'] = df['text'].str.count(r'[\'"]')

# Average sentence length
df['avg_sentence_length'] = df['word_count'] / df['sentence_count']

# Count emojis (basic regex)
df['emoji_count'] = df['text'].str.count(r'[\U0001F300-\U0001F9FF]')

# %%

df.drop(columns=['text'], inplace=True)
# %%
df.to_csv("../../data/interim/text-analysis/brunch_reviews_text_analysis_lite.csv", index=False)
# %%
