import os
# Force software rendering and off-screen mode for headless environments
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['PYVISTA_USE_IPYVTK'] = 'true'
os.environ['DISPLAY'] = ':99' 

import pyvista as pv
import numpy as np
import argparse
import imageio.v2 as imageio
import glob
from tqdm import tqdm


def render_frames(pred_root, gt_root, save_path, modes, total_steps=50):
    viz_config = {
        'Displacement': {'cmap': 'turbo', 'vmax': 300, 'unit': '(mm)'},
        'Plastic_Strain': {'cmap': 'plasma', 'vmax': 0.05, 'unit': '(strain)'},
        'Von_Mises_Stress': {'cmap': 'jet', 'vmax': 1, 'unit': '(MPa)'}
    }

    # Camera: match TemporalMesh pattern — use PyVista preset then reset_camera(bounds).
    # Options: "iso" (isometric), "xy", "xz", "yz", or list [(pos), (focal), (up)].
    ref_camera_position = "iso"

    for mode in modes:
        mode_save_path = os.path.join(save_path, mode)
        os.makedirs(mode_save_path, exist_ok=True)
        config = viz_config.get(mode, viz_config['Displacement'])

        # --- PRE-LOAD MESHES ONCE PER RUN ---
        loaded_meshes = {"Truth": {}, "Prediction": {}}
        for i in range(3):
            run_file = f"run_{i+1:03d}.vtp"
            gt_file_path = os.path.join(gt_root, run_file)
            pred_file_path = os.path.join(pred_root, run_file)
            
            if os.path.exists(gt_file_path):
                loaded_meshes["Truth"][i] = pv.read(gt_file_path)
            if os.path.exists(pred_file_path):
                loaded_meshes["Prediction"][i] = pv.read(pred_file_path)

        for step in tqdm(range(total_steps), desc=f"Rendering 3D {mode}"):
            # Window size increased to accommodate 6 full-height colorbars
            plotter = pv.Plotter(shape=(2, 3), window_size=[3000, 1600], off_screen=True)
            plotter.set_background("white")

            render_groups = [("Truth", 0), ("Prediction", 1)]
            
            for i in range(3):
                for label, row in render_groups:
                    # Retrieve the pre-loaded mesh
                    mesh = loaded_meshes[label].get(i)
                    plotter.subplot(row, i)
                    
                    if mesh is not None:
                        try:
                            # Deform mesh so positions actually change over time (base + displacement at this step)
                            base_xyz = np.array(mesh.points)
                            disp_array_name = f"displacement_t{step:03d}"
                            displacement = mesh.point_data.get(disp_array_name)
                            if displacement is not None:
                                displacement = np.asarray(displacement)
                                if displacement.ndim == 1:
                                    displacement = displacement.reshape(-1, 1).repeat(3, axis=1)
                                mesh_to_plot = mesh.copy(deep=True)
                                mesh_to_plot.points = base_xyz + displacement
                            else:
                                mesh_to_plot = mesh

                            # Scalars for coloring (e.g. "plastic_strain_t000")
                            array_name = f"{mode.lower()}_t{step:03d}"
                            raw_scalars = mesh.point_data.get(array_name)
                            if raw_scalars is None:
                                scalars = np.zeros(mesh_to_plot.n_points)
                            else:
                                raw_scalars = np.asarray(raw_scalars)
                                if mode == 'Displacement' and raw_scalars.ndim > 1:
                                    scalars = np.linalg.norm(raw_scalars, axis=1)
                                else:
                                    scalars = raw_scalars

                            # 1. Add mesh (deformed geometry so positions change; disable automatic scalar bar)
                            actor = plotter.add_mesh(
                                mesh_to_plot, scalars=scalars, cmap=config['cmap'],
                                clim=[0, config['vmax']], smooth_shading=True, lighting=True,
                                show_scalar_bar=False
                            )

                            # 2. FORCE INDIVIDUAL COLORBAR PER SUBPLOT
                            plotter.add_scalar_bar(
                                title=config['unit'],
                                mapper=actor.mapper,
                                vertical=True,
                                position_x=0.85, # Pushed right within the subplot
                                position_y=0.1,
                                height=0.8,
                                width=0.05,
                                label_font_size=18,
                                title_font_size=20,
                                color='black',
                                fmt="%.1f"
                            )
                            
                            # 3. Large Title with Mode and Step
                            title_str = f"{label} Run {i+1} | {mode.replace('_', ' ')} | Step {step}"
                            plotter.add_text(title_str, font_size=18, color="black")
                            
                            # 4. 3D Camera: iso + reset_camera; then flip 180° around Z (mirror X and Y)
                            plotter.camera_position = ref_camera_position
                            plotter.reset_camera(bounds=mesh_to_plot.bounds)
                            pos, focal, up = plotter.camera_position
                            pos, focal = np.array(pos), np.array(focal)
                            vec = pos - focal
                            new_pos = focal + np.array([-vec[0], -vec[1], vec[2]])
                            plotter.camera_position = [tuple(new_pos), tuple(focal), up]
                            plotter.add_axes()
                            plotter.camera.zoom(1.1)
                            
                        except Exception as e:
                            # If an error occurs, it will print to the console for debugging
                            print(f"Error at {label} Run {i+1} Step {step}: {e}")
                            plotter.add_text(f"Read Error", font_size=15, color='red')
                    else:
                        plotter.add_text(f"MISSING DATA", font_size=15, color="gray")

            plotter.screenshot(os.path.join(mode_save_path, f"frame_{step:03d}.png"))
            plotter.close()

        create_gif(mode_save_path, mode, save_path)

def create_gif(frame_dir, mode, save_root):
    print(f"Creating GIF for {mode}")
    output_gif = os.path.join(save_root, f'crash_{mode.lower()}.gif')
    search_path = os.path.join(frame_dir, "frame_*.png")
    frame_files = sorted(glob.glob(search_path))
    if not frame_files: 
        print(f"No frames found in {frame_dir}")
        return
    
    with imageio.get_writer(output_gif, mode='I', fps=10, loop=0) as writer:
        for f in frame_files:
            image = imageio.imread(f)
            writer.append_data(image)
            
    print(f"Created looping GIF at: {output_gif}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_root', type=str, required=True)
    parser.add_argument('--gt_root', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--modes', type=str, nargs='+', 
                        default=['Displacement', 'Plastic_Strain', 'Von_Mises_Stress'])
    args = parser.parse_args()

    # Path Validation
    for name, path in [("Prediction Root", args.pred_root), ("Ground Truth Root", args.gt_root)]:
        if not os.path.exists(path):
            print(f"ERROR: {name} path does not exist: '{path}'")
            exit(1)

    render_frames(args.pred_root, args.gt_root, args.save_path, args.modes)