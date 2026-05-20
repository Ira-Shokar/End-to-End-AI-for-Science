import json
from pathlib import Path
import itertools
import numpy as np

def print_record_info(record, label):
    print(f"{label}: {record.__class__.__name__}")
    for k, v in record.__dict__.items():
        if len(str(v)) > 200 or isinstance(v, np.ndarray):
            k = f"{k} (originally {type(v).__name__})"
            try:
                v = np.array([el for el in v])
                v = f"{v.shape} with range ({v.min():.3f}, {v.max():.3f})"
            except: 
                v = str(v)[:100] + "..."
        print(f" - {k}: {v}")

def preview_simulation_files(rad_path, json_path, start_line=1000, num_lines=30):
    """
    Prints a specific snippet of a .rad file and the contents of a .json file.
    """
    # --- Print snippet of the .rad file ---
    print(f"=== Snippet of {rad_path.name} (Lines {start_line} to {start_line + num_lines - 1}) ===")
    try:
        with open(rad_path, 'r') as rad_file:
            # islice efficiently skips ahead to start_line, then grabs the next num_lines
            rad_snippet = "".join(itertools.islice(rad_file, start_line, start_line + num_lines))
            
            if rad_snippet:
                print(rad_snippet)
                print("... [File truncated for display] ...\n")
            else:
                print(f"The file has fewer than {start_line} lines.\n")
                
    except FileNotFoundError:
        print("File not found. Please check the path.\n")

    # --- Print the contents of the runX.json file ---
    print(f"=== Contents of {json_path.name} ===")
    try:
        with open(json_path, 'r') as json_file:
            # Load and pretty-print the JSON
            run_data = json.load(json_file)
            print(json.dumps(run_data, indent=4))
    except FileNotFoundError:
        print("File not found. Please check the path.\n")