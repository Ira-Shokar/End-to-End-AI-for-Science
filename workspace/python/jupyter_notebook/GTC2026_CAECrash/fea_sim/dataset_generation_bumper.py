import os
import re
import json
import csv
import itertools
import shutil
from datetime import datetime

# ==============================================================================
# 1. CORE MODIFICATION FUNCTION
# ==============================================================================
def modify_radioss_file(input_path, output_path, 
                        geo_scales=(1.0, 1.0, 1.0), 
                        thick_scale=1.0, 
                        velocity_vector=None, 
                        rwall_updates=None):
    """
    Reads a Radioss .rad file and modifies it based on specific parameters.
    """
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

    sx, sy, sz = geo_scales
    
    # Flags and Counters
    processing_nodes = False
    processing_prop = False
    processing_inivel = False
    processing_rwall = False
    
    prop_line_count = 0 
    inivel_line_count = 0
    rwall_data_line_count = 0
    
    rwall_shift = [0.0, 0.0, 0.0] 

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        
        for line in f_in:
            sline = line.strip()
            
            # --- KEYWORD DETECTION ---
            if sline.startswith('/'):
                processing_nodes = False
                processing_prop = False
                processing_inivel = False
                processing_rwall = False
                
                prop_line_count = 0
                inivel_line_count = 0
                rwall_data_line_count = 0
                
                if sline.upper().startswith('/NODE'):
                    processing_nodes = True
                elif sline.upper().startswith('/PROP/SHELL'):
                    processing_prop = True
                elif sline.upper().startswith('/INIVEL'):
                    processing_inivel = True
                elif sline.upper().startswith('/RWALL'):
                    processing_rwall = True
                
                f_out.write(line)
                continue
            
            # --- IGNORE COMMENTS ---
            if not sline or sline.startswith('#'):
                f_out.write(line)
                continue

            # A. RIGID WALL (RWALL)
            if processing_rwall and rwall_updates:
                rwall_data_line_count += 1
                clean_line = sline.split('#')[0].strip()
                tokens = re.split(r'[,\s]+', clean_line)
                tokens = [t for t in tokens if t]

                if rwall_data_line_count == 3 and 'diameter' in rwall_updates:
                    if len(tokens) >= 3:
                        new_dia = rwall_updates['diameter']
                        tokens[2] = f"{new_dia:>20.12g}"
                        sb = [f"{t:>20}" for t in tokens]
                        f_out.write("".join(sb) + "\n")
                    else: f_out.write(line)

                elif rwall_data_line_count == 4 and 'origin' in rwall_updates:
                    if len(tokens) >= 3:
                        old_x, old_y, old_z = float(tokens[0]), float(tokens[1]), float(tokens[2])
                        new_x, new_y, new_z = rwall_updates['origin']
                        rwall_shift = [new_x - old_x, new_y - old_y, new_z - old_z]
                        sb = [f"{new_x:>20.12g}", f"{new_y:>20.12g}", f"{new_z:>20.12g}"]
                        f_out.write("".join(sb) + "\n")
                    else: f_out.write(line)

                elif rwall_data_line_count == 5 and 'origin' in rwall_updates:
                    if len(tokens) >= 3:
                        ox, oy, oz = float(tokens[0]), float(tokens[1]), float(tokens[2])
                        nx, ny, nz = ox + rwall_shift[0], oy + rwall_shift[1], oz + rwall_shift[2]
                        sb = [f"{nx:>20.12g}", f"{ny:>20.12g}", f"{nz:>20.12g}"]
                        f_out.write("".join(sb) + "\n")
                    else: f_out.write(line)
                else:
                    f_out.write(line)

            # B. NODAL SCALING
            elif processing_nodes:
                clean_line = sline.split('#')[0].strip()
                tokens = re.split(r'[,\s]+', clean_line)
                tokens = [t for t in tokens if t]
                try:
                    if len(tokens) == 4:
                        nid = tokens[0]
                        nx = float(tokens[1]) * sx
                        ny = float(tokens[2]) * sy
                        nz = float(tokens[3]) * sz
                        f_out.write(f"{nid:>10}{nx:>20.12g}{ny:>20.12g}{nz:>20.12g}\n")
                    elif len(tokens) == 3:
                        nx = float(tokens[0]) * sx
                        ny = float(tokens[1]) * sy
                        nz = float(tokens[2]) * sz
                        f_out.write(f"{nx:>20.12g}{ny:>20.12g}{nz:>20.12g}\n")
                    else:
                        f_out.write(line)
                except ValueError:
                    f_out.write(line)

            # C. THICKNESS SCALING
            elif processing_prop:
                prop_line_count += 1
                if prop_line_count == 4:
                    clean_line = sline.split('#')[0].strip()
                    tokens = re.split(r'[,\s]+', clean_line)
                    tokens = [t for t in tokens if t]
                    if len(tokens) >= 3:
                        try:
                            old_thick = float(tokens[2])
                            new_thick = old_thick * thick_scale
                            tokens[2] = f"{new_thick:.4f}"
                            sb = [f"{t:>20}" if i==2 else f"{t:>10}" for i,t in enumerate(tokens)]
                            f_out.write("".join(sb) + "\n")
                        except ValueError: f_out.write(line)
                    else: f_out.write(line)
                else:
                    f_out.write(line)

            # D. VELOCITY (INIVEL)
            elif processing_inivel and velocity_vector:
                inivel_line_count += 1
                if inivel_line_count == 2:
                    clean_line = sline.split('#')[0].strip()
                    tokens = re.split(r'[,\s]+', clean_line)
                    tokens = [t for t in tokens if t]
                    if len(tokens) >= 3:
                        vx, vy, vz = velocity_vector
                        sb = [f"{vx:>20.12g}", f"{vy:>20.12g}", f"{vz:>20.12g}"]
                        for t in tokens[3:]: sb.append(f"{t:>10}")
                        f_out.write("".join(sb) + "\n")
                    else: f_out.write(line)
                else: f_out.write(line)

            # E. DEFAULT
            else:
                f_out.write(line)

# ==============================================================================
# 2. HELPER: SUMMARY WRITER
# ==============================================================================
def save_summaries(dataset_dir, all_runs_data):
    """
    Saves summary.json and summary.csv to the dataset root.
    """
    # 1. Save JSON
    json_path = os.path.join(dataset_dir, "summary.json")
    with open(json_path, 'w') as f:
        json.dump(all_runs_data, f, indent=4)
    print(f"Summary JSON saved to: {json_path}")

    # 2. Save CSV
    csv_path = os.path.join(dataset_dir, "summary.csv")
    
    # Flatten data for CSV
    # We unpack tuples (X,Y,Z) into separate columns for easier Excel analysis
    flattened_data = []
    
    for run in all_runs_data:
        p = run['parameters']
        
        # Handle Geometry
        gx, gy, gz = p.get('geometry_scale', (1.0, 1.0, 1.0))
        
        # Handle Velocity (might be None)
        vel = p.get('velocity_vector')
        if vel:
            vx, vy, vz = vel
        else:
            vx, vy, vz = ("Default", "Default", "Default")

        # Handle Wall Origin (might be None)
        w_orig = p.get('rwall_origin')
        if w_orig:
            wx, wy, wz = w_orig
        else:
            wx, wy, wz = ("Default", "Default", "Default")

        row = {
            "Run_ID": run['run_id'],
            "Timestamp": run['timestamp'],
            "Geo_Scale_X": gx,
            "Geo_Scale_Y": gy,
            "Geo_Scale_Z": gz,
            "Vel_X": vx,
            "Vel_Y": vy,
            "Vel_Z": vz,
            "Thickness_Factor": p.get('thickness_scale', 1.0),
            "Wall_Diameter": p.get('rwall_diameter', "Default"),
            "Wall_X": wx,
            "Wall_Y": wy,
            "Wall_Z": wz
        }
        flattened_data.append(row)

    if flattened_data:
        headers = flattened_data[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(flattened_data)
        print(f"Summary CSV saved to: {csv_path}")

# ==============================================================================
# 3. DATASET GENERATION LOGIC
# ==============================================================================
def generate_dataset(base_file, engine_file, output_root, variations):
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created dataset directory: {output_root}")

    # Extract ranges
    geo_range = variations.get('geometry_scales', [(1.0, 1.0, 1.0)])
    vel_range = variations.get('velocities', [None])
    thick_range = variations.get('thickness_scales', [1.0])
    dia_range = variations.get('rwall_diameters', [None])
    origin_range = variations.get('rwall_origins', [None])
    
    # Cartesian Product
    combinations = list(itertools.product(geo_range, vel_range, thick_range, dia_range, origin_range))
    total_runs = len(combinations)
    
    print(f"\n==========================================")
    print(f" TOTAL CASES TO GENERATE: {total_runs}")
    print(f"==========================================\n")
    
    if not os.path.exists(engine_file):
        print(f"WARNING: Engine file '{engine_file}' not found! It will not be copied.")

    # List to hold data for summary files
    all_runs_summary = []

    for i, combo in enumerate(combinations, 1):
        geo, vel, thick, dia, origin = combo
        
        run_id = f"run{i}"
        run_dir = os.path.join(output_root, run_id)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            
        # 1. Modify the 0000.rad file
        filename_0000 = os.path.basename(base_file)
        out_file_path = os.path.join(run_dir, filename_0000)
        
        rwall_cfg = {}
        if dia is not None: rwall_cfg['diameter'] = dia
        if origin is not None: rwall_cfg['origin'] = origin
        if not rwall_cfg: rwall_cfg = None

        modify_radioss_file(
            input_path=base_file,
            output_path=out_file_path,
            geo_scales=geo,
            thick_scale=thick,
            velocity_vector=vel,
            rwall_updates=rwall_cfg
        )
        
        # 2. Copy the 0001.rad file
        if os.path.exists(engine_file):
            shutil.copy(engine_file, run_dir)
        
        # 3. Create Metadata Dictionary
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "geometry_scale": geo,
                "velocity_vector": vel,
                "thickness_scale": thick,
                "rwall_diameter": dia,
                "rwall_origin": origin
            }
        }
        
        # Save individual JSON
        json_path = os.path.join(run_dir, f"{run_id}.json")
        with open(json_path, 'w') as f_json:
            json.dump(metadata, f_json, indent=4)
        
        # Add to summary list
        all_runs_summary.append(metadata)
            
        print(f"[{i}/{total_runs}] {run_id} | Geo:{geo} | Vel:{vel} | Thk:{thick} | Wall:{origin}")

    # 4. Generate Summary Files
    print("\n--- GENERATING SUMMARIES ---")
    save_summaries(output_root, all_runs_summary)

# ==============================================================================
# 4. USER CONFIGURATION
# ==============================================================================
if __name__ == "__main__":
    
    # --- INPUT FILES ---
    STARTER_FILE = "Bumper_Beam_AP_meshed_0000.rad" 
    ENGINE_FILE = "Bumper_Beam_AP_meshed_0001.rad"

    # --- OUTPUT DIRECTORY ---
    DATASET_DIR = "dataset"

    # --- EXPERIMENT RANGES ---
    experiment_setup = {
        # 1. Geometry Scale (X, Y, Z)
        "geometry_scales": [
            (1.0, 1.0, 1.0),
            (1.0, 0.5, 1.0),
            (1.0, 1.0, 0.5),
            (1.0, 2.0, 1.0),
            (1.0, 1.0, 2.0),
        ],

        # 2. Velocity (Vx, Vy, Vz)
        "velocities": [
            (-5.0, 0.0, 0.0),
            (-3.0, 0.0, 0.0),
            (-7.0, 0.0, 0.0),
        ],
        
        # 3. Thickness Scaling Factor
        "thickness_scales": [
            1.0,
            0.7,
            1.3,
        ],
        
        # 4. Rigid Wall Diameter (mm)
        "rwall_diameters": [
            254.0
        ],
        
        # 5. Rigid Wall Origin (X, Y, Z)
        "rwall_origins": [
            (-170.0, 0.0, 0.0),
            (-170.0, 120.0, 0.0),
            (-170.0, 240.0, 0.0)
        ]
    }

    # --- EXECUTE ---
    try:
        generate_dataset(STARTER_FILE, ENGINE_FILE, DATASET_DIR, experiment_setup)
        print("\nDataset generation complete.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")