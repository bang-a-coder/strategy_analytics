import pandas as pd
import os
from sentiment_analysis import analyze_sentiment

# Create a sample Excel file with reviews including a long one
sample_data = {
    'Review': [
        "This product is amazing! I love it so much.",
        "I'm very disappointed with this service.",
        # Long review (>512 tokens)
        "I've been using this product for over a year now, and I have to say my experience has been mixed. " * 20 +
        "On one hand, the customer service is excellent and responsive. " * 10 +
        "However, the product quality has declined significantly in recent months. " * 10
    ]
}

# Create sample Excel file
sample_file = "sample_reviews.xlsx"
output_file = "sample_results.xlsx"

print("Creating sample Excel file...")
df = pd.DataFrame(sample_data)
df.to_excel(sample_file, index=False)
print(f"Sample file created with {len(df)} reviews")

# Run sentiment analysis
print("\nRunning sentiment analysis...")
analyze_sentiment(sample_file, output_file, "Review")

# Display results
print("\nResults:")
results = pd.read_excel(output_file)
for i, row in results.iterrows():
    print(f"Review {i+1} ({len(row['Review'].split())} words): {row['sentiment']} (score: {row['sentiment_score']:.4f})")

# Clean up
print("\nCleaning up...")
os.remove(sample_file)
os.remove(output_file)
print("Done!") 