#%%
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# %%

# df_feelings = pd.read_csv("../../data/processed/feelings/feelings_results_parsed.csv")
# df_lite = pd.read_csv("../../data/interim/text-analysis/brunch_reviews_text_analysis_lite.csv")
# df_full = pd.read_csv("../../data/interim/text-analysis/brunch_reviews_text_analysis.csv")
# # %%

# df_lite.head()

# # %%

# df_feelings = df_feelings.add_prefix('feeling_')

# # %%

# df_lite = pd.concat([df_lite, df_feelings], axis=1)

# # %%
# # df_lite.to_csv("../../data/processed/brunch_reviews_text_analysis_lite.csv", index=False)
# # %%
# df_feelings.columns
# # %%
# df_attributes = pd.read_csv("../../data/processed/business_details_with_attributes.csv")
# # %%

# df_reviews_with_business_details = pd.read_csv("../../data/interim/brunch_reviews_with_business_details.csv")

# # %%
# df_reviews_with_business_details.drop(columns=['text'], inplace=True)
# %%

df = pd.read_csv("../../data/processed/brunch_reviews_text_analysis_lite.csv")
# %%
df['stars_binary'] = df['stars'].apply(lambda x: "High" if x >= 4 else "Low")
# %%
df.to_csv("../../data/processed/brunch_reviews_text_analysis_lite.csv", index=False)
# %%
