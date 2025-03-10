import ujson
import csv
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

def process_chunk(chunk_data):
    """Process a chunk of JSON lines"""
    try:
        return [ujson.loads(line) for line in chunk_data if line.strip()]
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return []

def read_in_chunks(file, chunk_size):
    """Generator to read file in chunks"""
    chunk = []
    for i, line in enumerate(file):
        chunk.append(line)
        if (i + 1) % chunk_size == 0:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def json_to_csv(json_file, csv_file, chunk_size=10000):
    # Initialize headers
    with open(json_file, 'r') as f:
        first_line = f.readline()
        headers = ujson.loads(first_line).keys()
        
    # Write headers
    with open(csv_file, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()
    
    # Process in parallel
    n_workers = max(1, mp.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        with open(json_file, 'r') as infile, open(csv_file, 'a', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=headers)
            
            # Skip first line as we already read it
            next(infile)
            
            # Process chunks
            for chunk_results in executor.map(process_chunk, read_in_chunks(infile, chunk_size)):
                if chunk_results:
                    writer.writerows(chunk_results)

if __name__ == '__main__':
    # Example usage
    json_to_csv(
        './Data/yelp_academic_dataset_review.json',
        "./yelp_academic_dataset_review.csv",
        chunk_size=10000
    )