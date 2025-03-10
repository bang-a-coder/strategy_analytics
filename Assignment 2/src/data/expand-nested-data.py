
#%%

import pandas as pd
import numpy as np
from ast import literal_eval

import matplotlib.pyplot as plt
import seaborn as sns
import json
# %%

business_details = pd.read_csv("../../data/processed/business_details_with_attributes.csv")

reviews_data = pd.read_csv("../../data/processed/brunch_reviews_english.csv")

# %%
# Merge reviews with business details based on business_id
# expanded_reviews = reviews_data.merge(business_details, on='business_id', how='left')

# # %%
# grace_ids = pd.read_csv("./Cache/brunch_data.csv")
# # %%
# expanded_reviews.to_csv("./Cache/brunch_reviews_expanded.csv", index=False)
# %%



# %%
business_details['attributes'] = business_details['attributes'].fillna('{}')
business_details['attributes'] = business_details['attributes'].apply(literal_eval)

expanded_attributes = pd.json_normalize(business_details['attributes'])
business_details = pd.concat([business_details.drop('attributes', axis=1), expanded_attributes], axis=1)

# %%
# business_details.to_csv("./Data/Generated/business_details_with_attributes.csv")
# %%
