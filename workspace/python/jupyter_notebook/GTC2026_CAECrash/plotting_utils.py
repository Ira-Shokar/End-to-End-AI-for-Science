"""Plotting utilities for crash simulation analysis."""

import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_l2_errors(predictions, ground_truths, save_path='l2_error_plot.png'):
    """
    Compute and plot L2 errors from predictions and ground truths.

    Args:
        predictions: List of [T, N, 3] tensors (denormalized)
        ground_truths: List of [T, N, 3] tensors (denormalized)
        save_path: Path to save the plot

    Returns:
        Dictionary with error statistics
    """
    if len(predictions) == 0:
        print("No predictions found. Run inference first!")
        return None

    num_samples = len(predictions)
    T = predictions[0].shape[0]
    timesteps = np.arange(T)

    all_l2_errors = []
    all_relative_errors = []

    for sample_idx in range(num_samples):
        pred = predictions[sample_idx].numpy() if hasattr(predictions[sample_idx], 'numpy') else predictions[sample_idx]
        gt = ground_truths[sample_idx].numpy() if hasattr(ground_truths[sample_idx], 'numpy') else ground_truths[sample_idx]

        sample_l2_errors = []
        sample_rel_errors = []

        for t in range(T):
            error = np.linalg.norm(pred[t] - gt[t], axis=1)
            l2_error = np.mean(error)
            gt_norm = np.linalg.norm(gt[t], axis=1)
            rel_error = np.mean(error / (gt_norm + 1e-8)) * 100
            sample_l2_errors.append(l2_error)
            sample_rel_errors.append(rel_error)

        all_l2_errors.append(sample_l2_errors)
        all_relative_errors.append(sample_rel_errors)

    all_l2_errors = np.array(all_l2_errors)
    all_relative_errors = np.array(all_relative_errors)

    mean_l2 = np.mean(all_l2_errors, axis=0)
    std_l2 = np.std(all_l2_errors, axis=0)
    mean_rel = np.mean(all_relative_errors, axis=0)
    std_rel = np.std(all_relative_errors, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for sample_idx in range(all_l2_errors.shape[0]):
        axes[0].plot(timesteps, all_l2_errors[sample_idx], color='gray', alpha=0.3, linewidth=1)
    axes[0].plot(timesteps, mean_l2, 'o-', color='red', linewidth=2.5, markersize=6, label='Mean', zorder=10)
    axes[0].fill_between(timesteps, mean_l2 - std_l2, mean_l2 + std_l2, color='red', alpha=0.2, label='±1 std')
    axes[0].set_xlabel('Timestep', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('L2 Error (mm)', fontsize=12, fontweight='bold')
    axes[0].set_title('Absolute L2 Error Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    for sample_idx in range(all_relative_errors.shape[0]):
        axes[1].plot(timesteps, all_relative_errors[sample_idx], color='gray', alpha=0.3, linewidth=1)
    axes[1].plot(timesteps, mean_rel, 's-', color='red', linewidth=2.5, markersize=6, label='Mean', zorder=10)
    axes[1].fill_between(timesteps, mean_rel - std_rel, mean_rel + std_rel, color='red', alpha=0.2, label='±1 std')
    axes[1].set_xlabel('Timestep', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Relative L2 Error Over Time', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'timesteps': timesteps,
        'all_l2_errors': all_l2_errors,
        'all_relative_errors': all_relative_errors,
        'mean_l2': mean_l2, 'std_l2': std_l2,
        'mean_rel': mean_rel, 'std_rel': std_rel,
    }


def load_vtp_data(vtp_file_path: str):
    """
    Load a VTP file and extract all available fields.

    Args:
        vtp_file_path: Path to the VTP file.

    Returns:
        Dictionary with mesh data, point fields, and boundary conditions.
    """
    from vtk.util import numpy_support
    from physicsnemo.datapipes.cae.readers import read_vtp

    mesh = read_vtp(vtp_file_path)
    points = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    point_data = mesh.GetPointData()
    field_names = [
        point_data.GetArray(i).GetName()
        for i in range(point_data.GetNumberOfArrays())
    ]

    disp_keys = [k for k in field_names if k.startswith("displacement_t")]
    disp_keys.sort(key=lambda k: float(k.split('_t')[-1]))
    first_field, last_field = disp_keys[0], disp_keys[-1]

    thick_vals = (
        numpy_support.vtk_to_numpy(point_data.GetArray("thickness"))
        if "thickness" in field_names else None
    )
    pstrain_last = (
        numpy_support.vtk_to_numpy(point_data.GetArray("plastic_strain"))
        if "plastic_strain" in field_names else None
    )
    stress_last = (
        numpy_support.vtk_to_numpy(point_data.GetArray("von_mises_stress"))
        if "von_mises_stress" in field_names else None
    )

    field_data = mesh.GetFieldData()
    bc_keys = [
        field_data.GetArray(i).GetName()
        for i in range(field_data.GetNumberOfArrays())
        if field_data.GetArray(i).GetName().startswith("bc_")
    ]

    return {
        "mesh": mesh, "points": points,
        "x": x, "y": y, "z": z,
        "point_data": point_data, "field_names": field_names,
        "disp_keys": disp_keys, "first_field": first_field, "last_field": last_field,
        "thick_vals": thick_vals, "pstrain_last": pstrain_last, "stress_last": stress_last,
        "bc_keys": bc_keys, "field_data": field_data,
    }


def print_vtp_summary(vtp_data):
    """Print a summary of loaded VTP data.

    Args:
        vtp_data: Dictionary returned by load_vtp_data().
    """
    from vtk.util import numpy_support

    points = vtp_data["points"]
    disp_keys = vtp_data["disp_keys"]
    first_field = vtp_data["first_field"]
    last_field = vtp_data["last_field"]
    thick_vals = vtp_data["thick_vals"]
    pstrain_last = vtp_data["pstrain_last"]
    stress_last = vtp_data["stress_last"]
    bc_keys = vtp_data["bc_keys"]
    field_names = vtp_data["field_names"]
    field_data = vtp_data["field_data"]

    print(f"\n{'='*60}")
    print("VTP DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Points:     {points.shape[0]:,} nodes")
    print(f"Timesteps:  {len(disp_keys)} (from {first_field} to {last_field})")
    print(f"Fields:     {len(field_names)} total")
    print(f"\n--- Available Data ---")
    print(f"  Displacement:    {len(disp_keys)} timesteps")
    print(f"  Thickness:       {'Available' if thick_vals is not None else 'Not found'}")
    print(f"  Plastic Strain:  {'Available' if pstrain_last is not None else 'Not in VTP (update VTP sink)'}")
    print(f"  Von Mises Stress: {'Available' if stress_last is not None else 'Not in VTP (update VTP sink)'}")
    if bc_keys:
        print(f"  Boundary Conds:  {len(bc_keys)} keys")
    else:
        print(f"  Boundary Conds:  Not found (update VTP sink + configure boundary_condition_keys)")

    if bc_keys:
        print(f"\n--- Boundary Conditions ---")
        for key in bc_keys:
            bc_val = numpy_support.vtk_to_numpy(field_data.GetArray(key))
            print(f"  {key}: {bc_val}")


def plot_vtp_multifield(vtp_data, subsample_step=1):
    """Visualize displacement, thickness, plastic strain & stress from VTP data.

    Args:
        vtp_data: Dictionary returned by load_vtp_data().
        subsample_step: Point subsampling factor (increase for faster rendering).
    """
    from vtk.util import numpy_support

    x, y, z = vtp_data["x"], vtp_data["y"], vtp_data["z"]
    point_data = vtp_data["point_data"]
    first_field = vtp_data["first_field"]
    last_field = vtp_data["last_field"]
    thick_vals = vtp_data["thick_vals"]
    pstrain_last = vtp_data["pstrain_last"]
    stress_last = vtp_data["stress_last"]
    step = subsample_step

    def get_warped_data(field_name):
        disp_vec = numpy_support.vtk_to_numpy(point_data.GetArray(field_name))
        mag = np.linalg.norm(disp_vec, axis=1)
        return x + disp_vec[:, 0], y + disp_vec[:, 1], z + disp_vec[:, 2], mag

    x0, y0, z0, mag0 = get_warped_data(first_field)
    x1, y1, z1, mag1 = get_warped_data(last_field)
    disp_clim = (0, max(mag0.max(), mag1.max()))

    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

    def setup_axis(ax, title):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    num_plots = 2 + (1 if thick_vals is not None else 0) + \
                    (1 if pstrain_last is not None else 0) + \
                    (1 if stress_last is not None else 0)
    ncols, nrows = 3, (num_plots + 2) // 3
    fig = plt.figure(figsize=(16, 5.5 * nrows))
    plot_idx = 1

    ax1 = fig.add_subplot(nrows, ncols, plot_idx, projection='3d')
    p1 = ax1.scatter(x0[::step], y0[::step], z0[::step], c=mag0[::step],
                     cmap='jet', s=3, vmin=disp_clim[0], vmax=disp_clim[1], alpha=0.8)
    setup_axis(ax1, f"Displacement at Start\n{first_field}")
    fig.colorbar(p1, ax=ax1, shrink=0.6, label='Displacement (mm)', pad=0.1)
    plot_idx += 1

    ax2 = fig.add_subplot(nrows, ncols, plot_idx, projection='3d')
    p2 = ax2.scatter(x1[::step], y1[::step], z1[::step], c=mag1[::step],
                     cmap='jet', s=3, vmin=disp_clim[0], vmax=disp_clim[1], alpha=0.8)
    setup_axis(ax2, f"Displacement at End\n{last_field}")
    fig.colorbar(p2, ax=ax2, shrink=0.6, label='Displacement (mm)', pad=0.1)
    plot_idx += 1

    if thick_vals is not None:
        ax3 = fig.add_subplot(nrows, ncols, plot_idx, projection='3d')
        p3 = ax3.scatter(x[::step], y[::step], z[::step], c=thick_vals[::step],
                         cmap='viridis', s=3, alpha=0.8)
        setup_axis(ax3, "Shell Thickness\n(Static Property)")
        fig.colorbar(p3, ax=ax3, shrink=0.6, label='Thickness (mm)', pad=0.1)
        plot_idx += 1

    if pstrain_last is not None:
        ax4 = fig.add_subplot(nrows, ncols, plot_idx, projection='3d')
        p4 = ax4.scatter(x1[::step], y1[::step], z1[::step], c=pstrain_last[::step],
                         cmap='plasma', s=3, vmin=0, alpha=0.8)
        setup_axis(ax4, f"Plastic Strain (End)\n{last_field}")
        fig.colorbar(p4, ax=ax4, shrink=0.6, label='Effective Plastic Strain', pad=0.1)
        plot_idx += 1
        print(f"  Plastic strain range: [{pstrain_last.min():.6f}, {pstrain_last.max():.6f}]")

    if stress_last is not None:
        ax5 = fig.add_subplot(nrows, ncols, plot_idx, projection='3d')
        p5 = ax5.scatter(x1[::step], y1[::step], z1[::step], c=stress_last[::step],
                         cmap='hot', s=3, vmin=0, alpha=0.8)
        setup_axis(ax5, f"Von Mises Stress (End)\n{last_field}")
        fig.colorbar(p5, ax=ax5, shrink=0.6, label='Stress (MPa)', pad=0.1)
        plot_idx += 1
        print(f"  Stress range: [{stress_last.min():.2f}, {stress_last.max():.2f}] MPa")

    plt.suptitle("Crash Simulation: Multi-Field Visualization", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()

    print(f"\nVisualization complete: {num_plots} plots generated")


def save_trajectory_to_vtp(positions_with_physics, output_dir, run_id):
    """
    Save a trajectory sequence to VTP files.
    positions_with_physics: [T, N, 5] tensor (XYZ + Strain + Stress)
    """
    T, N, C = positions_with_physics.shape
    os.makedirs(f"{output_dir}/run_{run_id:03d}", exist_ok=True)

    for t in range(T):
        data = positions_with_physics[t].numpy()
        xyz = data[:, :3]
        cloud = pv.PolyData(xyz)
        if C >= 4:
            cloud.point_data["Plastic_Strain"] = data[:, 3]
        if C >= 5:
            cloud.point_data["Von_Mises_Stress"] = data[:, 4]
        cloud.save(f"{output_dir}/run_{run_id:03d}/timestep_{t:04d}.vtp")
