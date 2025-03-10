import json
import csv
import sys
from multiprocessing import Pool, cpu_count
import os

def process_chunk(chunk):
    return [{k: str(v) for k, v in item.items()} for item in chunk]

def json_to_csv(json_file, csv_file, chunk_size=10000):
    try:
        file_size = os.path.getsize(json_file)
        use_streaming = file_size > 100 * 1024 * 1024  # Stream if > 100MB
        
        with open(json_file, 'r') as f:
            # For small files, process directly
            if not use_streaming:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
                headers = list(data[0].keys())
                
                with open(csv_file, 'w', newline='') as out:
                    writer = csv.DictWriter(out, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(process_chunk(data))
                return
            
            # For large files, stream and process in parallel
            with open(csv_file, 'w', newline='') as out:
                # Read first object to get headers
                first_char = f.read(1)
                if not first_char:
                    raise ValueError("Empty JSON file")
                f.seek(0)
                
                if first_char == '[':
                    first_obj = next(json.JSONDecoder().raw_decode(f.read(4096))[0])
                else:
                    first_obj = json.loads(first_char + f.read(4096))
                
                headers = list(first_obj.keys())
                writer = csv.DictWriter(out, fieldnames=headers)
                writer.writeheader()
                
                # Process in parallel chunks
                with Pool(cpu_count()) as pool:
                    f.seek(0)
                    decoder = json.JSONDecoder()
                    buffer = []
                    
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                            
                        try:
                            obj, idx = decoder.raw_decode(chunk)
                            if isinstance(obj, list):
                                buffer.extend(obj)
                            else:
                                buffer.append(obj)
                                
                            if len(buffer) >= chunk_size:
                                processed = pool.map(process_chunk, [buffer])[0]
                                writer.writerows(processed)
                                buffer = []
                        except json.JSONDecodeError:
                            continue
                    
                    if buffer:
                        processed = pool.map(process_chunk, [buffer])[0]
                        writer.writerows(processed)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_csv.py input.json output.csv")
        sys.exit(1)
    
    json_to_csv(sys.argv[1], sys.argv[2])
    print(f"Conversion complete: {sys.argv[1]} → {sys.argv[2]}") 