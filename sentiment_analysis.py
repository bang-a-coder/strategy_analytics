import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import argparse

def analyze_sentiment(input_file, output_file, text_column):
    # Load data
    df = pd.read_excel(input_file)
    
    # Load model and tokenizer for long text handling
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Function to handle long texts by chunking and averaging
    def analyze_long_text(text):
        if not isinstance(text, str) or not text.strip():
            return {"label": "NEUTRAL", "score": 0.5}
            
        # Tokenize the text
        tokens = tokenizer.tokenize(text)
        
        # If text is short enough, process normally
        if len(tokens) <= 512:
            result = sentiment_analyzer(text)[0]
            return result
            
        # For long texts, split into chunks and average results
        chunks = []
        chunk_size = 500  # Slightly less than max to account for special tokens
        
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i+chunk_size]
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
        
        # Process each chunk
        results = sentiment_analyzer(chunks)
        
        # Calculate weighted average based on chunk length
        pos_score = sum(r['score'] if r['label'] == 'POSITIVE' else (1-r['score']) 
                        for r in results) / len(results)
        
        # Determine final sentiment
        if pos_score > 0.5:
            return {"label": "POSITIVE", "score": pos_score}
        else:
            return {"label": "NEGATIVE", "score": 1 - pos_score}
    
    # Initialize sentiment pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    # Process reviews
    print(f"Processing {len(df)} reviews...")
    results = []
    
    for text in df[text_column]:
        results.append(analyze_long_text(text))
    
    # Add results to dataframe
    df['sentiment_score'] = [result['score'] for result in results]
    df['sentiment'] = [result['label'] for result in results]
    
    # Save results
    df.to_excel(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis for Long Reviews")
    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--output", required=True, help="Path to output Excel file")
    parser.add_argument("--column", required=True, help="Name of the column containing review text")
    
    args = parser.parse_args()
    analyze_sentiment(args.input, args.output, args.column) 