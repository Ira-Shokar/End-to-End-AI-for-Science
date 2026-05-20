import os
import sys
import subprocess
import glob
import concurrent.futures
import time

from vortex_radioss.animtod3plot.Anim_to_D3plot import readAndConvert

# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================

# --- PATHS ---
OPENRADIOSS_ROOT = os.environ.get("OPENRADIOSS_ROOT")
DATASET_DIR = "dataset" 

# --- EXECUTABLES ---
STARTER_EXE = os.path.join(OPENRADIOSS_ROOT, "exec/starter_linux64_gf")
ENGINE_EXE  = os.path.join(OPENRADIOSS_ROOT, "exec/engine_linux64_gf")
TO_VTK_EXE  = os.path.join(OPENRADIOSS_ROOT, "exec/anim_to_vtk_linux64_gf")

# --- SIMULATION SETTINGS ---
INPUT_BASE_NAME = "Bumper_Beam_AP_meshed" 
STARTER_FILE    = f"{INPUT_BASE_NAME}_0000.rad"
ENGINE_FILE     = f"{INPUT_BASE_NAME}_0001.rad"

# --- COMPUTATIONAL RESOURCES ---
# Warning: Ensure (MAX_PARALLEL_JOBS * OMP_NUM_THREADS) <= Total CPU Cores
MAX_PARALLEL_JOBS = 2
OMP_NUM_THREADS   = "8" 

# --- DEBUG MODE ---
DEBUG_MODE = False

# ==============================================================================
# 2. ENVIRONMENT SETUP
# ==============================================================================
def get_radioss_env():
    """
    Constructs the environment variables dictionary required by OpenRadioss.
    """
    env = os.environ.copy()
    
    env["OPENRADIOSS_PATH"] = OPENRADIOSS_ROOT
    env["RAD_CFG_PATH"] = os.path.join(OPENRADIOSS_ROOT, "hm_cfg_files")
    env["OMP_STACKSIZE"] = "400m"
    env["OMP_NUM_THREADS"] = OMP_NUM_THREADS
    
    lib_paths = [
        os.path.join(OPENRADIOSS_ROOT, "extlib/hm_reader/linux64/"),
        os.path.join(OPENRADIOSS_ROOT, "extlib/h3d/lib/linux64/")
    ]
    
    current_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{':'.join(lib_paths)}:{current_ld}"
    
    # Python path might be needed for the vortex library if it's not installed globally
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") 
    
    return env

# ==============================================================================
# 3. HELPER: D3PLOT CONVERTER
# ==============================================================================
def run_d3plot_conversion(cwd, stem_name, env, log_file):
    """
    Runs the vortex_radioss library in a subprocess to convert Anim to D3plot.
    """
    # --- FIX IS HERE ---
    # We must construct the ABSOLUTE path. 
    # If we pass a relative path, Lasso tries to write temp files to root '/'.
    abs_stem_path = os.path.abspath(os.path.join(cwd, stem_name))
    
    # Python one-liner to execute the library
    python_cmd = (
        f"from vortex_radioss.animtod3plot.Anim_to_D3plot import readAndConvert; "
        f"readAndConvert('{abs_stem_path}')"
    )
    
    # We use sys.executable to ensure we use the same python interpreter
    cmd = [sys.executable, "-c", python_cmd]
    
    # Note: We still run in 'cwd' to capture logs there, but the Python script
    # now works with a full absolute path for the data.
    subprocess.run(cmd, cwd=cwd, env=env, stdout=log_file, stderr=log_file, check=True)
    
# ==============================================================================
# 4. WORKER FUNCTION (Runs one case)
# ==============================================================================

from tqdm import tqdm
    
def run_case(run_folder, pbar):
    """
    Executes Starter -> Engine -> VTK -> D3PLOT in the given folder.
    """
    case_id = os.path.basename(run_folder)
    log_prefix = f"[{case_id}]"

    tqdm.write(f"{log_prefix} Starting processing in: {run_folder}")

    try:    
        env = get_radioss_env()
        cwd = run_folder
    
        # --- STEP 1: STARTER ---
        starter_cmd = [STARTER_EXE, "-i", STARTER_FILE, "-nt", OMP_NUM_THREADS]
        
        try:
            with open(os.path.join(cwd, "starter.log"), "w") as log:
                subprocess.run(starter_cmd, cwd=cwd, env=env, stdout=log, stderr=log, check=True)
        except subprocess.CalledProcessError:
            print(f"{log_prefix} FAIL: Starter. See starter.log")
            return False
    
        # --- STEP 2: ENGINE ---
        engine_cmd = [ENGINE_EXE, "-i", ENGINE_FILE]
        
        try:
            with open(os.path.join(cwd, "engine.log"), "w") as log:
                subprocess.run(engine_cmd, cwd=cwd, env=env, stdout=log, stderr=log, check=True)
            print(f"{log_prefix} Simulation Complete.")
        except subprocess.CalledProcessError:
            print(f"{log_prefix} FAIL: Engine. See engine.log")
            return False
    
        # --- CHECK FOR ANIMATION FILES ---
        # We look for the first animation file (A001) to verify we have data
        first_anim = os.path.join(cwd, f"{INPUT_BASE_NAME}A001")
        if not os.path.exists(first_anim):
            print(f"{log_prefix} Warning: No animation files (A001) found. Skipping conversions.")
            return True
    
        # --- STEP 3: ANIM TO VTK ---
        try:
            anim_files = sorted(glob.glob(os.path.join(cwd, f"{INPUT_BASE_NAME}A[0-9][0-9][0-9]")))
            if anim_files:
                for a_file in anim_files:
                    fname = os.path.basename(a_file)
                    vtk_path = os.path.join(cwd, f"{fname}.vtk")
                    with open(vtk_path, "w") as f_out:
                        subprocess.run([TO_VTK_EXE, fname], cwd=cwd, env=env, stdout=f_out, check=True)
        except Exception as e:
            print(f"{log_prefix} Error in VTK conversion: {e}")
    
        # --- STEP 4: ANIM TO D3PLOT ---
        try:
            # print(f"{log_prefix} Converting to D3PLOT...")
            with open(os.path.join(cwd, "d3plot_conv.log"), "w") as log:
                run_d3plot_conversion(cwd, INPUT_BASE_NAME, env, log)
            print(f"{log_prefix} D3PLOT Generated.")
        except Exception as e:
            print(f"{log_prefix} Error in D3PLOT conversion: {e} (See d3plot_conv.log)")
        tqdm.write(f"{log_prefix} Completed successfully")
        return True

    except Exception as e:
        tqdm.write(f"{log_prefix} ❌ Failed: {e}")
        return False

    finally:
        # Count this case as finished (success or failure — your choice)
        pbar.update(1)

# ==============================================================================
# 5. MAIN ORCHESTRATOR
# ==============================================================================
if __name__ == "__main__":
    
    # 1. Discover Run Folders
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        exit(1)
        
    all_runs = [f.path for f in os.scandir(DATASET_DIR) if f.is_dir() and "run" in f.name]
    all_runs.sort()

    if not all_runs:
        print("No run folders found.")
        exit()

    # 2. Debug Mode Check
    if DEBUG_MODE:
        print(f"DEBUG MODE ON: Running only 1 case ({all_runs[0]})")
        run_case(all_runs[0])
        exit()

    # 3. Parallel Execution
    print(f"--- STARTING BATCH EXECUTION ---")
    print(f"Cases: {len(all_runs)} | Jobs: {MAX_PARALLEL_JOBS} | Threads/Job: {OMP_NUM_THREADS}")
    
    start_time = time.time()

    pbar = tqdm(
        total=len(all_runs),
        desc="Running Radioss cases",
        unit="case",
        dynamic_ncols=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        future_to_run = {executor.submit(run_case, folder, pbar): folder for folder in all_runs}
        
        for future in concurrent.futures.as_completed(future_to_run):
            folder = future_to_run[future]
            try:
                future.result()
            except Exception as e:
                print(f"Exception in {folder}: {e}")

    elapsed = time.time() - start_time
    print(f"BATCH COMPLETE. Total time: {elapsed:.2f} seconds.")