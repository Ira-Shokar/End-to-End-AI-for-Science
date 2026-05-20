"""Utilities for inspecting D3plot and OpenRadioss RAD files."""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from lasso.dyna import D3plot, ArrayType


def compute_von_mises_stress(stress_tensor: np.ndarray) -> np.ndarray:
    """Calculate Von Mises equivalent stress from 6-component stress tensor.

    Handles various shapes from d3plot files:
    - (timesteps, num_elements, 6)
    - (timesteps, num_elements, integration_points, 6)
    - (timesteps, num_elements, layers, integration_points, 6)

    For multi-point data, averages over integration points/layers.

    Args:
        stress_tensor: Stress tensor with last dimension = 6 components
                      [σxx, σyy, σzz, τxy, τyz, τzx]

    Returns:
        Von Mises stress array with shape (timesteps, num_elements)
    """
    # Average over integration points/layers if present
    while stress_tensor.ndim > 3:
        stress_tensor = np.mean(stress_tensor, axis=2)

    # Extract components: [σxx, σyy, σzz, τxy, τyz, τzx]
    sxx = stress_tensor[..., 0]
    syy = stress_tensor[..., 1]
    szz = stress_tensor[..., 2]
    sxy = stress_tensor[..., 3]
    syz = stress_tensor[..., 4]
    szx = stress_tensor[..., 5]

    # Von Mises formula
    von_mises = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3 * (sxy**2 + syz**2 + szx**2)
    )
    return von_mises


def compute_node_field_from_elements(
    element_data: np.ndarray,
    mesh_connectivity: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """Map element-centered data to nodes via weighted averaging.

    Each node's value is the average of all connected elements.
    """
    num_timesteps = element_data.shape[0]
    node_field_all_timesteps = np.zeros((num_timesteps, num_nodes), dtype=np.float64)

    # Detect indexing offset (1-based vs 0-based)
    conn_min = np.min(mesh_connectivity)
    conn_max = np.max(mesh_connectivity)
    offset = -1 if (conn_max >= num_nodes and conn_min >= 1) else 0

    for t in range(num_timesteps):
        element_data_t = element_data[t, :]
        node_field_sum = np.zeros(num_nodes)
        node_field_count = np.zeros(num_nodes)

        for i, element in enumerate(mesh_connectivity):
            field_value = element_data_t[i]
            for node_idx in element:
                idx = int(node_idx + offset)
                if 0 <= idx < num_nodes:
                    node_field_sum[idx] += field_value
                    node_field_count[idx] += 1

        mask = node_field_count > 0
        node_field_sum[mask] /= node_field_count[mask]
        node_field_all_timesteps[t, :] = node_field_sum

    return node_field_all_timesteps


def parse_thickness_from_rad(rad_file_path: str) -> dict:
    """
    Parse thickness data from OpenRadioss *_0000.rad file.
    Uses header-based column alignment to find 'Thick' values accurately.
    """
    result = {
        'property_thickness': {},
        'node_thickness': {},
        'all_values': [],
        'source': None
    }
    
    if not os.path.exists(rad_file_path):
        print(f"⚠️ RAD file not found: {rad_file_path}")
        return result
    
    print(f"📂 Parsing thickness from: {os.path.basename(rad_file_path)}")
    
    with open(rad_file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith("/PROP/SHELL"):
            parts = line.split('/')
            try:
                prop_id = int(parts[3]) if len(parts) > 3 else None
            except (IndexError, ValueError):
                prop_id = None
            
            if prop_id is None:
                i += 1
                continue
            
            found_thick = False
            for offset in range(1, 25):
                if i + offset >= len(lines):
                    break
                
                scan_line = lines[i + offset].strip()
                
                if scan_line.startswith("/") and not scan_line.startswith("#"):
                    break
                
                if scan_line.startswith("#") and "Thick" in scan_line:
                    data_line_index = i + offset + 1
                    if data_line_index < len(lines):
                        data_line = lines[data_line_index].strip()
                        
                        headers = re.split(r'\s{2,}', scan_line.lstrip('#').strip())
                        values = re.split(r'\s+', data_line)
                        
                        thick_idx = -1
                        for idx, h in enumerate(headers):
                            if "Thick" in h:
                                thick_idx = idx
                                break
                        
                        if thick_idx != -1 and thick_idx < len(values):
                            try:
                                thickness = float(values[thick_idx])
                                result['property_thickness'][prop_id] = thickness
                                result['all_values'].append(thickness)
                                result['source'] = 'PROP/SHELL'
                                found_thick = True
                                print(f"   Property {prop_id}: Thick = {thickness:.4f} mm")
                            except ValueError:
                                pass
                        break
            
            if not found_thick:
                print(f"   Property {prop_id}: ⚠️ Thick not found")
        
        elif line.startswith('/THIC'):
            result['source'] = 'THIC'
            i += 1
            while i < len(lines):
                data_line = lines[i].strip()
                if data_line.startswith('/') and not data_line.startswith('#'):
                    break
                if not data_line or data_line.startswith('#'):
                    i += 1
                    continue
                
                values = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', data_line)
                if len(values) >= 2:
                    try:
                        node_id = int(float(values[0]))
                        thickness = float(values[1])
                        result['node_thickness'][node_id] = thickness
                        result['all_values'].append(thickness)
                    except ValueError:
                        pass
                i += 1
            continue
        
        i += 1
    
    seen = set()
    unique_values = []
    for v in result['all_values']:
        if v not in seen:
            seen.add(v)
            unique_values.append(v)
    result['all_values'] = unique_values
    
    print(f"\n   📋 Found {len(result['property_thickness'])} properties with thickness")
    
    return result


def find_rad_file(d3plot_path: str) -> str:
    """Find the corresponding *_0000.rad file for a d3plot."""
    base_dir = os.path.dirname(d3plot_path)
    
    patterns = [
        os.path.join(base_dir, "*_0000.rad"),
        os.path.join(base_dir, "*.rad"),
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            for match in matches:
                if '_0000.rad' in match:
                    return match
            return matches[0]
    
    return None


def plot_histogram(ax, data, title, xlabel, color, bins=50):
    """Helper to plot a styled histogram."""
    ax.hist(data.flatten(), bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics annotation
    stats_text = f"Min: {np.min(data):.4f}\nMax: {np.max(data):.4f}\nMean: {np.mean(data):.4f}\nStd: {np.std(data):.4f}"
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def inspect_and_plot_d3plot(data_path: str):
    """
    Load metadata, generate detailed stats, correct for absolute coordinates,
    and visualize training variable distributions with histograms.
    """
    if not os.path.exists(data_path):
        print(f"❌ Error: File not found at {data_path}")
        return

    print(f"🔄 Loading: {os.path.basename(data_path)} ...")
    
    dp = D3plot(data_path)
    
    # 1. GATHER DATA & STATS
    if 'timesteps' in dp.arrays:
        sim_times = dp.arrays['timesteps']
    elif 'time' in dp.arrays:
        sim_times = dp.arrays['time']
    else:
        print("⚠️ Warning: Time array not found. Using step indices.")
        temp_disp = dp.arrays['node_displacement']
        sim_times = np.arange(temp_disp.shape[0])

    coords = dp.arrays['node_coordinates']
    num_nodes = coords.shape[0]
    
    num_shells = dp.arrays['element_shell_node_indexes'].shape[0] if 'element_shell_node_indexes' in dp.arrays else 0
    num_solids = dp.arrays['element_solid_node_indexes'].shape[0] if 'element_solid_node_indexes' in dp.arrays else 0
    mesh_connectivity = dp.arrays['element_shell_node_indexes'] if 'element_shell_node_indexes' in dp.arrays else None

    raw_data = dp.arrays['node_displacement']

    # 2. PRINT DETAILED SUMMARY
    print(f"\n📊 --- Data Summary for {os.path.basename(data_path)} ---")
    print(f"  • Time Steps:       {len(sim_times)} frames")
    print(f"  • Nodes:            {num_nodes:,}")
    print(f"  • Elements:         {num_shells:,} Shells, {num_solids:,} Solids")
    print(f"  • Dimensions:       {raw_data.shape} (Time, Nodes, XYZ)")

    print(f"\n🔍 --- Available Data Fields ---")
    keys = list(dp.arrays.keys())
    print(f"  • Keys found: {keys[:15]} ... (+ {max(0, len(keys)-15)} more)")

    # 3. ABSOLUTE COORDINATES CORRECTION
    initial_magnitude = np.linalg.norm(raw_data[0], axis=1)
    max_initial_val = np.max(initial_magnitude)

    print(f"\n🔍 Data Check at T=0.0s: Max value is {max_initial_val:.2f} mm")

    if max_initial_val > 1.0:
        print("   ⚠️  DETECTED ABSOLUTE COORDINATES.")
        print("   🛠️  Applying Fix: Subtracting initial position.")
        true_displacement = raw_data - raw_data[0]
    else:
        print("   ✅ Data is relative displacement. No fix needed.")
        true_displacement = raw_data

    # 4. COMPUTE DISPLACEMENT
    disp_magnitude = np.linalg.norm(true_displacement, axis=2)
    max_disp_over_time = np.max(disp_magnitude, axis=1)

    print(f"\n📈 --- Displacement Analysis Result ---")
    print(f"   • Max Deformation: {np.max(max_disp_over_time):.2f} mm")

    # 5. EXTRACT PLASTIC STRAIN (if available)
    plastic_strain_node = None
    if ArrayType.element_shell_effective_plastic_strain in dp.arrays:
        print(f"\n✅ Extracting Plastic Strain...")
        plastic_strain_elem = dp.arrays[ArrayType.element_shell_effective_plastic_strain]
        
        # Average over integration points if needed
        while plastic_strain_elem.ndim > 2:
            plastic_strain_elem = np.mean(plastic_strain_elem, axis=2)
        
        if mesh_connectivity is not None:
            plastic_strain_node = compute_node_field_from_elements(
                plastic_strain_elem, mesh_connectivity, num_nodes
            )
            print(f"   • Shape: {plastic_strain_node.shape}")
            print(f"   • Range: [{np.min(plastic_strain_node):.6f}, {np.max(plastic_strain_node):.6f}]")
    else:
        print(f"\n⚠️ Plastic strain not available in d3plot")

    # 6. EXTRACT VON MISES STRESS (if available)
    von_mises_node = None
    if ArrayType.element_shell_stress in dp.arrays:
        print(f"\n✅ Extracting Von Mises Stress...")
        stress_tensor = dp.arrays[ArrayType.element_shell_stress]
        von_mises_elem = compute_von_mises_stress(stress_tensor)
        
        # Average over integration points if needed
        while von_mises_elem.ndim > 2:
            von_mises_elem = np.mean(von_mises_elem, axis=2)
        
        if mesh_connectivity is not None:
            von_mises_node = compute_node_field_from_elements(
                von_mises_elem, mesh_connectivity, num_nodes
            )
            print(f"   • Shape: {von_mises_node.shape}")
            print(f"   • Range: [{np.min(von_mises_node):.2f}, {np.max(von_mises_node):.2f}] MPa")
    else:
        print(f"\n⚠️ Stress tensor not available in d3plot")

    # 7. THICKNESS ANALYSIS
    print(f"\n" + "="*50)
    print("📏 --- Thickness Analysis ---")
    print("="*50)
    
    rad_file = find_rad_file(data_path)
    thickness_data = None
    
    if rad_file:
        print(f"✅ Found RAD file: {os.path.basename(rad_file)}")
        thickness_data = parse_thickness_from_rad(rad_file)
        
        if thickness_data['all_values']:
            print(f"\n📊 --- Thickness Summary ---")
            print(f"   • Source:     {thickness_data['source']}")
            print(f"   • Properties: {len(thickness_data['property_thickness'])}")
            print(f"   • Min:        {np.min(thickness_data['all_values']):.3f} mm")
            print(f"   • Max:        {np.max(thickness_data['all_values']):.3f} mm")
            print(f"   • Mean:       {np.mean(thickness_data['all_values']):.3f} mm")
    else:
        print(f"⚠️ No *_0000.rad file found in {os.path.dirname(data_path)}")

    # ========================================
    # 8. VISUALIZATION: TIME SERIES + HISTOGRAMS
    # ========================================
    print(f"\n" + "="*50)
    print("📊 --- Training Variable Distributions ---")
    print("="*50)
    
    # Determine subplot layout based on available data
    num_histograms = 1  # displacement always available
    if thickness_data and thickness_data['all_values']:
        num_histograms += 1
    if plastic_strain_node is not None:
        num_histograms += 1
    if von_mises_node is not None:
        num_histograms += 1
    
    # Create figure with time series on top row, histograms on bottom
    fig = plt.figure(figsize=(14, 10))
    
    # --- TOP ROW: Time Series Plot ---
    ax_time = fig.add_subplot(2, 1, 1)
    ax_time.plot(sim_times, max_disp_over_time, 
                 color='#d62728', linewidth=2.5, label='Max Node Displacement')
    ax_time.set_xlabel('Simulation Time', fontsize=11)
    ax_time.set_ylabel('Displacement (mm)', fontsize=11)
    ax_time.set_title(f'Crash Dynamics: Maximum Displacement Over Time\nFile: {os.path.basename(data_path)}', 
                      fontsize=12, fontweight='bold')
    ax_time.grid(True, alpha=0.3, linestyle='--')
    ax_time.legend(loc='upper left')
    
    # Add secondary y-axis for stress/strain if available
    if plastic_strain_node is not None:
        ax_strain = ax_time.twinx()
        max_strain_over_time = np.max(plastic_strain_node, axis=1)
        ax_strain.plot(sim_times, max_strain_over_time, 
                       color='#2ca02c', linewidth=2, linestyle='--', label='Max Plastic Strain')
        ax_strain.set_ylabel('Plastic Strain [-]', fontsize=11, color='#2ca02c')
        ax_strain.tick_params(axis='y', labelcolor='#2ca02c')
        ax_strain.legend(loc='upper right')
    
    # --- BOTTOM ROW: Histograms ---
    axes_hist = []
    for i in range(num_histograms):
        ax = fig.add_subplot(2, num_histograms, num_histograms + i + 1)
        axes_hist.append(ax)
    
    hist_idx = 0
    
    # Histogram 1: Final Displacement Magnitude
    final_disp = disp_magnitude[-1, :]  # Last timestep
    plot_histogram(axes_hist[hist_idx], final_disp, 
                   'Node Displacement (Final Timestep)', 
                   'Displacement Magnitude (mm)', '#d62728')
    hist_idx += 1
    
    # Histogram 2: Thickness (if available)
    if thickness_data and thickness_data['all_values']:
        thickness_arr = np.array(thickness_data['all_values'])
        plot_histogram(axes_hist[hist_idx], thickness_arr,
                       'Shell Thickness Distribution',
                       'Thickness (mm)', '#1f77b4', bins=20)
        hist_idx += 1
    
    # Histogram 3: Plastic Strain (if available)
    if plastic_strain_node is not None:
        # Use final timestep, exclude zero values for better visualization
        final_strain = plastic_strain_node[-1, :]
        nonzero_strain = final_strain[final_strain > 1e-6]
        if len(nonzero_strain) > 0:
            plot_histogram(axes_hist[hist_idx], nonzero_strain,
                           'Plastic Strain (Final, Non-Zero)',
                           'Effective Plastic Strain [-]', '#2ca02c')
        else:
            plot_histogram(axes_hist[hist_idx], final_strain,
                           'Plastic Strain (Final Timestep)',
                           'Effective Plastic Strain [-]', '#2ca02c')
        hist_idx += 1
    
    # Histogram 4: Von Mises Stress (if available)
    if von_mises_node is not None:
        final_stress = von_mises_node[-1, :]
        nonzero_stress = final_stress[final_stress > 1e-3]
        if len(nonzero_stress) > 0:
            plot_histogram(axes_hist[hist_idx], nonzero_stress,
                           'Von Mises Stress (Final, Non-Zero)',
                           'Stress (MPa)', '#ff7f0e')
        else:
            plot_histogram(axes_hist[hist_idx], final_stress,
                           'Von Mises Stress (Final Timestep)',
                           'Stress (MPa)', '#ff7f0e')
    
    plt.tight_layout()
    
    output_file = "training_variables_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Combined analysis plot saved to: {output_file}")
    plt.show()
    
    # ========================================
    # 9. SUMMARY TABLE
    # ========================================
    print(f"\n" + "="*60)
    print("📋 --- TRAINING VARIABLES SUMMARY ---")
    print("="*60)
    print(f"{'Variable':<25} {'Shape':<20} {'Min':<12} {'Max':<12} {'Mean':<12}")
    print("-"*60)
    print(f"{'mesh_pos':<25} {str(raw_data.shape):<20} {np.min(raw_data):<12.2f} {np.max(raw_data):<12.2f} {np.mean(raw_data):<12.2f}")
    print(f"{'displacement (final)':<25} {str(final_disp.shape):<20} {np.min(final_disp):<12.4f} {np.max(final_disp):<12.4f} {np.mean(final_disp):<12.4f}")
    
    if thickness_data and thickness_data['all_values']:
        t_arr = np.array(thickness_data['all_values'])
        print(f"{'thickness':<25} {str(t_arr.shape):<20} {np.min(t_arr):<12.4f} {np.max(t_arr):<12.4f} {np.mean(t_arr):<12.4f}")
    
    if plastic_strain_node is not None:
        print(f"{'plastic_strain':<25} {str(plastic_strain_node.shape):<20} {np.min(plastic_strain_node):<12.6f} {np.max(plastic_strain_node):<12.6f} {np.mean(plastic_strain_node):<12.6f}")
    
    if von_mises_node is not None:
        print(f"{'von_mises_stress':<25} {str(von_mises_node.shape):<20} {np.min(von_mises_node):<12.2f} {np.max(von_mises_node):<12.2f} {np.mean(von_mises_node):<12.2f}")
    
    print("="*60)
