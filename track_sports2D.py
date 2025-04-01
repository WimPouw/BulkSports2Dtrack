#!/usr/bin/env python
"""
Sports2D Video Processing Script
===============================
This script processes videos using Sports2D to track human motion and compute joint angles.
It has been updated to work with Sports2D v0.7.3.
"""

import os
import glob
import pandas as pd
import re
import argparse
import sys
import traceback
from datetime import datetime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process videos with Sports2D')
    
    # Path arguments
    parser.add_argument('--input_folder', default='./ToRetrack/', help='Folder containing videos to process')
    parser.add_argument('--output_folder', default='./RetrackedSports2D/', help='Folder for processed videos')
    parser.add_argument('--timeseries_folder', default='./RetrackedSports2DTimeSeries/', help='Folder for time series data')
    parser.add_argument('--meta_folder', default='./BodyMetaInfo/', help='Folder containing participant metadata')
    
    # Processing options
    parser.add_argument('--model', default='body_with_feet', choices=['body_with_feet', 'wholebody'], 
                        help='Pose model to use')
    parser.add_argument('--mode', default='balanced', choices=['lightweight', 'balanced', 'performance'], 
                        help='Processing mode: faster but less accurate (lightweight) or more accurate but slower (performance)')
    parser.add_argument('--multiperson', action='store_true', default=True, 
                        help='Process multiple people in the video')
    parser.add_argument('--do_ik', action='store_true', default=False, 
                        help='Run inverse kinematics to generate OpenSim animations (requires OpenSim)')
    parser.add_argument('--use_gpu', action='store_true', default=False, 
                        help='Try to use GPU acceleration if available')
    
    # OpenSim options
    parser.add_argument('--visible_side', default='auto', choices=['auto', 'left', 'right', 'front', 'back'],
                        help='Visible side of the person (auto=automatic detection)')
    parser.add_argument('--participant_mass', type=float, default=None,
                        help='Participant mass in kg (only affects forces, not kinematics)')
    parser.add_argument('--use_detailed_model', action='store_true', default=False,
                        help='Use OpenSim model with contact spheres and muscles')
    parser.add_argument('--generate_animation', action='store_true', default=False,
                        help='Generate an OpenSim animation script (requires OpenSim)')
    
    # Output options
    parser.add_argument('--save_video', action='store_true', default=True, 
                        help='Save processed video')
    parser.add_argument('--save_images', action='store_true', default=False, 
                        help='Save individual processed frames')
    parser.add_argument('--show_realtime', action='store_true', default=False, 
                        help='Show processing in real-time')
    parser.add_argument('--no_progress', action='store_true', default=False,
                        help='Disable progress display in terminal')
    
    # Resume processing
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume processing from where it left off, skipping already processed videos')
    
    args = parser.parse_args()
    
    # Automatically enable do_ik if generate_animation is specified
    if args.generate_animation and not args.do_ik:
        print("Note: Enabling inverse kinematics because --generate_animation was specified")
        args.do_ik = True
        
    return args

# Parse command-line arguments
args = parse_arguments()

# Define input and output folders
input_folder = os.path.abspath(args.input_folder)
output_folder = os.path.abspath(args.output_folder)
output_folder_timeseries = os.path.abspath(args.timeseries_folder)
meta_folder = os.path.abspath(args.meta_folder)

def get_participant_weight(video_name):
    """Extract participant ID from video name and get their weight from meta file"""
    # Extract participant ID (assuming format like p1_trial_1_video_raw.mp4)
    match = re.search(r'p(\d+)', video_name)
    if not match:
        print(f"Could not extract participant ID from {video_name}, using default weight")
        return 70.0  # Default weight in kg
    
    participant_id = match.group(1)
    meta_file = os.path.join(meta_folder, f'bodymeta_p{participant_id}.csv')
    
    if not os.path.exists(meta_file):
        print(f"Metadata file {meta_file} not found, using default weight")
        return 70.0
    
    try:
        # Try first to read as a simple CSV file with semicolon delimiter
        try:
            df = pd.read_csv(meta_file, sep=';')
            if 'weight' in df.columns:
                weight_kg = float(str(df['weight'].iloc[0]).replace(',', '.'))
                print(f"Found weight {weight_kg} kg in the weight column")
                return weight_kg
        except Exception:
            pass
            
        # If the above fails, load the file content and try to extract the weight
        with open(meta_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Check if we have a header row with field names and data row(s) with values
        lines = content.split('\n')
        if len(lines) >= 2:  # At least header + one data row
            header = lines[0].split(';')
            data = lines[1].split(';')
            
            # Find the position of 'weight' in the header
            try:
                weight_pos = [i for i, field in enumerate(header) if 'weight' in field.lower()]
                if weight_pos:
                    weight_index = weight_pos[0]
                    if weight_index < len(data):
                        weight_kg = float(data[weight_index].replace(',', '.'))
                        print(f"Found weight {weight_kg} kg at position {weight_index}")
                        return weight_kg
            except Exception:
                pass
        
        # Special case: If it's a single line with header and values combined
        if len(lines) == 1 and ';' in lines[0]:
            header_data = lines[0].split(';')
            # Find a field that looks like "weight=XXX" or similar
            for field in header_data:
                if 'weight' in field.lower():
                    # Extract number after "="
                    weight_match = re.search(r'=\s*([0-9]+[.,]?[0-9]*)', field)
                    if weight_match:
                        weight_kg = float(weight_match.group(1).replace(',', '.'))
                        print(f"Found weight {weight_kg} kg in field {field}")
                        return weight_kg
        
        # If we still haven't found it, look for any number in the file that could be a weight
        # Weight values are typically between 40-150 kg for adults
        weight_matches = re.findall(r'[^0-9]([4-9][0-9](?:[.,][0-9]+)?|1[0-4][0-9](?:[.,][0-9]+)?)[^0-9]', content)
        for match in weight_matches:
            weight_kg = float(match.replace(',', '.'))
            if 40 <= weight_kg <= 150:  # Reasonable weight range in kg
                print(f"Found likely weight value of {weight_kg} kg")
                return weight_kg
        
        # Last resort: typically weight is the 4th value in the format
        all_values = re.findall(r'[0-9]+(?:[.,][0-9]+)?', content)
        if len(all_values) >= 4:  # If we have at least 4 numeric values
            potential_weight = float(all_values[3].replace(',', '.'))
            if 40 <= potential_weight <= 150:  # Check if in reasonable range
                print(f"Using 4th numeric value {potential_weight} as weight")
                return potential_weight
        
        # If we get here, we couldn't find the weight
        print(f"Could not extract weight from {meta_file}, using default weight")
        return 70.0
                
    except Exception as e:
        print(f"Error reading metadata file for weight {meta_file}: {e}")
        return 70.0  # Default weight in kg
    
def get_participant_height(video_name):
    """Extract participant ID from video name and get their height from meta file"""
    # Extract participant ID (assuming format like p1_trial_1_video_raw.mp4)
    match = re.search(r'p(\d+)', video_name)
    if not match:
        print(f"Could not extract participant ID from {video_name}, using default height")
        return 1.70  # Default height in meters
    
    participant_id = match.group(1)
    meta_file = os.path.join(meta_folder, f'bodymeta_p{participant_id}.csv')
    
    if not os.path.exists(meta_file):
        print(f"Metadata file {meta_file} not found, using default height")
        return 1.70
    
    try:
        # Based on the document info provided, we know the CSV has a single line with a header
        # Format appears to be: "ppn;sex;age;weight;height;underarmlength;upperarmlength;upperarmcircumference;upperarmfold;handedness"
        with open(meta_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Try first to read as a simple CSV file with semicolon delimiter
        try:
            df = pd.read_csv(meta_file, sep=';')
            if 'height' in df.columns:
                height_cm = float(str(df['height'].iloc[0]).replace(',', '.'))
                print(f"Found height {height_cm} cm in the height column")
                return height_cm / 100.0
        except Exception:
            pass
            
        # If the above fails, check if we have a header row with field names
        # and data row(s) with values
        lines = content.split('\n')
        if len(lines) >= 2:  # At least header + one data row
            header = lines[0].split(';')
            data = lines[1].split(';')
            
            # Find the position of 'height' in the header
            try:
                height_pos = [i for i, field in enumerate(header) if 'height' in field.lower()]
                if height_pos:
                    height_index = height_pos[0]
                    if height_index < len(data):
                        height_cm = float(data[height_index].replace(',', '.'))
                        print(f"Found height {height_cm} cm at position {height_index}")
                        return height_cm / 100.0
            except Exception:
                pass
        
        # Special case: If it's a single line with header and values combined
        # Format: "ppn;sex;age;weight;height;underarmlength;..."
        if len(lines) == 1 and ';' in lines[0]:
            header_data = lines[0].split(';')
            # Find a field that looks like "height=XXX" or similar
            for field in header_data:
                if 'height' in field.lower():
                    # Extract number after "="
                    height_match = re.search(r'=\s*([0-9]+[.,]?[0-9]*)', field)
                    if height_match:
                        height_cm = float(height_match.group(1).replace(',', '.'))
                        print(f"Found height {height_cm} cm in field {field}")
                        return height_cm / 100.0
        
        # If we still haven't found it, look for any number in the file that could be a height
        height_matches = re.findall(r'[^0-9]([0-9]{3}(?:[.,][0-9]+)?)[^0-9]', content)
        for match in height_matches:
            height_cm = float(match.replace(',', '.'))
            if 140 <= height_cm <= 220:  # Reasonable height range in cm
                print(f"Found likely height value of {height_cm} cm")
                return height_cm / 100.0
        
        # Last resort: simply extract the 5th value if it exists (based on typical format)
        all_values = re.findall(r'[0-9]+(?:[.,][0-9]+)?', content)
        if len(all_values) >= 5:  # If we have at least 5 numeric values
            potential_height = float(all_values[4].replace(',', '.'))
            if 140 <= potential_height <= 220:  # Check if in reasonable range
                print(f"Using 5th numeric value {potential_height} as height")
                return potential_height / 100.0
        
        # If we get here, we couldn't find the height
        print(f"Could not extract height from {meta_file}, using default height")
        print(f"File content: {content}")
        return 1.70
                
    except Exception as e:
        print(f"Error reading metadata file {meta_file}: {e}")
        return 1.70  # Default height in meters

def generate_opensim_animation(video_name, video_output_dir, ts_output_dir):
    """Generate a script to visualize OpenSim animation"""
    if not args.do_ik or not args.generate_animation:
        return
    
    # Find the model file
    model_files = glob.glob(os.path.join(video_output_dir, "*.osim"))
    if not model_files:
        print("No OpenSim model file found. Cannot generate animation script.")
        return
    
    model_file = model_files[0]
    
    # Find the motion file
    motion_files = glob.glob(os.path.join(video_output_dir, "angles*.mot"))
    if not motion_files:
        print("No OpenSim motion file found. Cannot generate animation script.")
        return
    
    motion_file = motion_files[0]
    
    # Create Python script for OpenSim visualization
    animate_script_path = os.path.join(ts_output_dir, f"{video_name}_animate_opensim.py")
    
    # This is a multiline string with proper triple quotes handling
    script_content = f'''#!/usr/bin/env python
# OpenSim Animation Script for {video_name}
# Generated by track_sports2d.py

import os
import sys
import opensim as osim
import time

def animate_model(model_file, motion_file, speed=1.0):
    """
    Animate an OpenSim model with the given motion file.
    
    Parameters:
    -----------
    model_file : str
        Path to the OpenSim model file (.osim)
    motion_file : str
        Path to the motion file (.mot)
    speed : float
        Playback speed factor (1.0 = real-time)
    """
    # Initialize OpenSim
    osim.Logger.setLevelString("error")
    
    # Load the model
    print(f"Loading model: {{os.path.basename(model_file)}}")
    model = osim.Model(model_file)
    model.finalizeConnections()
    
    # Create a visualizer
    try:
        viz = osim.Visualizer(model)
        viz.show()
        print("OpenSim Visualizer initialized")
    except Exception as e:
        print(f"Failed to initialize visualizer: {{e}}")
        print("Make sure you're running this script with the OpenSim GUI installed.")
        sys.exit(1)
    
    # Load the motion
    print(f"Loading motion: {{os.path.basename(motion_file)}}")
    motion = osim.Storage(motion_file)
    
    # Get time range
    initial_time = motion.getFirstTime()
    final_time = motion.getLastTime()
    
    # Apply motion to the model
    print(f"Motion duration: {{final_time-initial_time:.2f}} seconds")
    print("Starting animation...")
    
    # Create a state for the model
    state = model.initSystem()
    
    # Animate
    step_size = 1.0/30.0  # 30 fps
    current_time = initial_time
    
    try:
        while current_time <= final_time:
            start_real_time = time.time()
            
            # Set the model state at the current time
            motion.getDataAtTime(current_time, state.getNY(), 
                                 state.getY().getNativeDouble())
            
            # Update the visualizer
            viz.show(state)
            
            # Sleep to maintain animation speed
            elapsed = time.time() - start_real_time
            sleep_time = (step_size / speed) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Advance time
            current_time += step_size
        
        print("Animation completed.")
        
        # Keep visualizer open
        input("Press Enter to exit...")
        
    except KeyboardInterrupt:
        print("Animation interrupted.")

if __name__ == "__main__":
    # Model and motion file paths
    model_file = r"{os.path.abspath(model_file)}"
    motion_file = r"{os.path.abspath(motion_file)}"
    
    # Start animation
    animate_model(model_file, motion_file, speed=1.0)
'''
    
    # Write the script to file
    with open(animate_script_path, 'w') as f:
        f.write(script_content)
    
    print(f"OpenSim animation script generated: {animate_script_path}")
    print("Run this script with Python to visualize the OpenSim animation.")
    
    # Also create a simple batch/shell script to run it
    if os.name == 'nt':  # Windows
        batch_path = os.path.join(ts_output_dir, f"{video_name}_animate_opensim.bat")
        batch_content = f"""@echo off
echo Running OpenSim animation for {video_name}...
python "{os.path.abspath(animate_script_path)}"
pause
"""
        with open(batch_path, 'w') as f:
            f.write(batch_content)
        print(f"Windows batch file created: {batch_path}")
    else:  # Unix/Mac
        shell_path = os.path.join(ts_output_dir, f"{video_name}_animate_opensim.sh")
        shell_content = f"""#!/bin/bash
echo "Running OpenSim animation for {video_name}..."
python "{os.path.abspath(animate_script_path)}"
"""
        with open(shell_path, 'w') as f:
            f.write(shell_content)
        os.chmod(shell_path, 0o755)  # Make executable
        print(f"Shell script created: {shell_path}")

def process_video(video_path):
    """Process a single video file with Sports2D"""
    print(f"Processing: {video_path}")
    
    try:
        # Get the video file name and directory
        video_file = os.path.basename(video_path)
        video_dir = os.path.dirname(os.path.abspath(video_path))
        video_name = os.path.splitext(video_file)[0]
        
        # Create output directory for this video's results
        video_output_dir = os.path.join(output_folder, video_name)
        ts_output_dir = os.path.join(output_folder_timeseries, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        os.makedirs(ts_output_dir, exist_ok=True)
        
        # Get participant height and weight from metadata
        participant_height = get_participant_height(video_name)
        print(f"Using participant height: {participant_height:.2f} meters")
        
        # Get participant weight (either from metadata or from command-line argument)
        if args.participant_mass:
            participant_weight = args.participant_mass
            print(f"Using command-line specified participant weight: {participant_weight:.1f} kg")
        else:
            participant_weight = get_participant_weight(video_name)
            print(f"Using participant weight: {participant_weight:.1f} kg")
        
        # Configure backend based on GPU availability
        backend = 'auto'
        device = 'auto'
        if args.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    backend = 'cuda'
                    print("CUDA GPU detected and will be used for processing")
                else:
                    # Check for MPS on Mac
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        backend = 'mps'
                        print("Apple MPS (Metal) detected and will be used for processing")
                    else:
                        print("No GPU detected, using CPU")
            except ImportError:
                print("PyTorch not installed, using CPU")
        
        # Configure Sports2D parameters (updated format for v0.7.3)
        config = {
            'project': {
                'video_input': [video_file],           # List of video files
                'video_dir': video_dir,                # Directory containing videos
                'time_range': [],                      # Process entire video
                'px_to_m_person_height': participant_height,  # Use height from metadata
                'px_to_m_from_person_id': 1,           # Assuming first person is main subject
                'visible_side': [args.visible_side],   # User-specified visible side
                'load_trc_px': '',                     # No trc file to load
                'compare': False,                      # Don't compare
                'webcam_id': 0,                        # Not using webcam
                'input_size': [1280, 720]              # Default input resolution
            },
            'process': {
                'multiperson': args.multiperson,       # User-specified
                'show_realtime_results': args.show_realtime,
                'save_vid': args.save_video,
                'save_img': args.save_images,
                'save_pose': True,
                'calculate_angles': True,
                'save_angles': True,
                'result_dir': video_output_dir         # Output directory
            },
            'pose': {
                'slowmo_factor': 1,                    # Default slowmo factor
                'pose_model': args.model.lower(),      # User-specified model (lowercase)
                'mode': args.mode,                     # User-specified mode
                'det_frequency': 4,                    # Run detection every 4 frames
                'device': device,
                'backend': backend,
                'tracking_mode': 'sports2d',           # Default tracking mode
                'deepsort_params': "{'max_age':30, 'n_init':3, 'nms_max_overlap':0.8, 'max_cosine_distance':0.3, 'nn_budget':200, 'max_iou_distance':0.8, 'embedder_gpu': True}",
                'keypoint_likelihood_threshold': 0.3,  # Default threshold
                'average_likelihood_threshold': 0.5,   # Default threshold
                'keypoint_number_threshold': 0.3       # Default threshold
            },
            'px_to_meters_conversion': {
                'to_meters': True,                     # Convert pixels to meters
                'make_c3d': True,                      # Create C3D files
                'calib_file': '',                      # No calibration file
                'floor_angle': 'auto',                 # Auto-detect floor angle
                'xy_origin': ['auto'],                 # Auto-detect origin
                'save_calib': True                     # Save calibration
            },
            'angles': {
                'display_angle_values_on': ['body', 'list'],
                'fontSize': 0.3,                       # Default font size
                'joint_angles': ['Right ankle', 'Left ankle', 'Right knee', 'Left knee', 
                                'Right hip', 'Left hip', 'Right shoulder', 'Left shoulder', 
                                'Right elbow', 'Left elbow', 'Right wrist', 'Left wrist'],
                'segment_angles': ['Right foot', 'Left foot', 'Right shank', 'Left shank', 
                                  'Right thigh', 'Left thigh', 'Pelvis', 'Trunk', 'Shoulders', 
                                  'Head', 'Right arm', 'Left arm', 'Right forearm', 'Left forearm'],
                'flip_left_right': True,               # Flip angles for consistent values
                'correct_segment_angles_with_floor_angle': True  # Correct for floor angle
            },
            'post-processing': {
                'interpolate': True,                   # Interpolate missing data
                'interp_gap_smaller_than': 10,         # Default value
                'fill_large_gaps_with': 'last_value',  # Default value
                'filter': True,                        # Apply filtering
                'show_graphs': False,                  # Don't show graphs
                'filter_type': 'butterworth',          # Default filter
                'butterworth': {                       # Default parameters
                    'order': 4,
                    'cut_off_frequency': 3             # User-specified cut-off
                },
                'gaussian': {                          # Default parameters
                    'sigma_kernel': 1
                },
                'loess': {                             # Default parameters
                    'nb_values_used': 5
                },
                'median': {                            # Default parameters
                    'kernel_size': 3
                }
            },
            'kinematics': {
                'do_ik': args.do_ik,                   # User-specified
                'use_augmentation': args.do_ik,        # Use augmentation if doing IK
                'use_contacts_muscles': args.use_detailed_model,
                'participant_mass': participant_weight,  # Use weight from metadata
                'right_left_symmetry': True,           # Default symmetry
                'default_height': 1.7,                 # Default height
                'remove_individual_scaling_setup': True,
                'remove_individual_ik_setup': True,
                'fastest_frames_to_remove_percent': 0.1,
                'close_to_zero_speed_px': 50,
                'close_to_zero_speed_m': 0.2,
                'large_hip_knee_angles': 45,
                'trimmed_extrema_percent': 0.5,
                'osim_setup_path': '../OpenSim_setup'
            },
            'logging': {
                'use_custom_logging': not args.no_progress  # Enable detailed progress logging
            }
        }
        
        # Run Sports2D processing
        from Sports2D import Sports2D
        Sports2D.process(config)
        
        # Move time series files if needed
        import shutil
        
        # Copy TRC files (pose coordinates)
        trc_files = glob.glob(os.path.join(video_output_dir, "*.trc"))
        for file in trc_files:
            if os.path.exists(file):
                dest_file = os.path.join(ts_output_dir, os.path.basename(file))
                shutil.copy2(file, dest_file)
        
        # Copy MOT files (angle data)
        mot_files = glob.glob(os.path.join(video_output_dir, "*.mot"))
        for file in mot_files:
            if os.path.exists(file):
                dest_file = os.path.join(ts_output_dir, os.path.basename(file))
                shutil.copy2(file, dest_file)
                
        # Copy C3D files if created
        c3d_files = glob.glob(os.path.join(video_output_dir, "*.c3d"))
        for file in c3d_files:
            if os.path.exists(file):
                dest_file = os.path.join(ts_output_dir, os.path.basename(file))
                shutil.copy2(file, dest_file)
        
        # Copy OpenSim files if IK was performed
        if args.do_ik:
            osim_files = glob.glob(os.path.join(video_output_dir, "*.osim"))
            for file in osim_files:
                if os.path.exists(file):
                    dest_file = os.path.join(ts_output_dir, os.path.basename(file))
                    shutil.copy2(file, dest_file)
            
            # Generate OpenSim animation script if requested
            if args.generate_animation:
                generate_opensim_animation(video_name, video_output_dir, ts_output_dir)
        
        print(f"Completed processing: {video_path}")
        print(f"Output video: {os.path.join(video_output_dir, f'{video_name}_tracked.mp4')}")
        print(f"Output time series: {ts_output_dir}")
        return True
        
    except Exception as e:
        print(f"Error processing {os.path.basename(video_path)}: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_welcome_message():
    """Print welcome message and usage instructions"""
    welcome = """
==========================================================================
                     Sports2D Video Processing Tool
==========================================================================

This script uses Sports2D to track human motion in videos and compute joint
positions and angles. Results are saved as processed videos and data files
that can be used for biomechanical analysis.

Basic Usage:
-----------
1. Place your videos in the './ToRetrack/' folder
2. Place participant metadata CSV files in './BodyMetaInfo/' folder
   (should be named like 'bodymeta_p1.csv', 'bodymeta_p2.csv', etc.)
3. Run this script: python track_sports2d.py

Advanced Options:
---------------
- Run with GPU:               python track_sports2d.py --use_gpu
- Faster processing:          python track_sports2d.py --mode lightweight --show_realtime False
- Run inverse kinematics:     python track_sports2d.py --do_ik
  (requires OpenSim, see installation instructions)

OpenSim Animation:
----------------
- Generate OpenSim animation: python track_sports2d.py --do_ik --generate_animation
  This will create animation scripts to visualize the tracked motion in OpenSim
- Specify visible side:        python track_sports2d.py --do_ik --visible_side left
  Options: auto, left, right, front, back

For a full list of options:  python track_sports2d.py --help
==========================================================================
"""
    print(welcome)

def check_dependencies():
    """Check if required dependencies are installed"""
    # Check if Sports2D is installed
    try:
        from Sports2D import Sports2D
        print("✓ Sports2D is installed")
    except ImportError:
        print("\n❌ Sports2D is not installed. Please install it with:")
        print("\npip install sports2d\n")
        print("For more information, visit: https://github.com/davidpagnon/sports2d")
        return False
    
    # Check if pandas is installed
    try:
        import pandas
        print("✓ pandas is installed")
    except ImportError:
        print("\n❌ pandas is not installed. Please install it with:")
        print("\npip install pandas\n")
        return False
    
    # Check for OpenSim if inverse kinematics is requested
    if args.do_ik:
        try:
            import opensim
            print("✓ OpenSim Python API is installed")
        except ImportError:
            print("\n❌ OpenSim Python API is not installed, but required for inverse kinematics.")
            print("To install OpenSim with conda:")
            print("\nconda install -c opensim-org opensim\n")
            print("For more details, see: https://github.com/davidpagnon/sports2d#full-install")
            return False
    
    # Check for GPU support if requested
    if args.use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("✓ Apple MPS (Metal) GPU detected")
            else:
                print("⚠ No compatible GPU detected. Will use CPU instead.")
            
            try:
                import onnxruntime as ort
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    print("✓ ONNX Runtime with CUDA support is installed")
                elif 'CoreMLExecutionProvider' in ort.get_available_providers():
                    print("✓ ONNX Runtime with CoreML support is installed")
                else:
                    print("⚠ ONNX Runtime is installed but without GPU support")
            except ImportError:
                print("⚠ ONNX Runtime with GPU support is not installed")
                print("To install with GPU support: pip install onnxruntime-gpu")
        except ImportError:
            print("⚠ PyTorch is not installed. Will use CPU only.")
            print("To enable GPU support, install PyTorch: pip install torch torchvision")
    
    return True

def main():
    """Main function to process all video files"""
    # Print welcome message
    print_welcome_message()
    
    # Check dependencies
    if not check_dependencies():
        print("\nMissing dependencies. Please install them and try again.")
        sys.exit(1)
    
    # Create required directories if they don't exist
    for directory in [input_folder, output_folder, output_folder_timeseries, meta_folder]:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)
    
    # Check if metadata directory exists and has files
    meta_files = glob.glob(os.path.join(meta_folder, '*.csv'))
    if not meta_files:
        print(f"Warning: No CSV files found in {meta_folder}. Will use default heights.")
    else:
        print(f"Found {len(meta_files)} participant metadata files")
    
    # Get all video files in the input folder
    video_files = glob.glob(os.path.join(input_folder, '*.mp4'))
    video_files.extend(glob.glob(os.path.join(input_folder, '*.avi')))  # Also check for AVI files
    
    if not video_files:
        print(f"No video files found in {input_folder}")
        print("Please place your videos in the ToRetrack folder")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Process each video file
    successful = 0
    failed = 0
    skipped = 0
    
    for i, video_path in enumerate(video_files):
        print(f"\n{'='*50}")
        print(f"Processing video {i+1}/{len(video_files)}: {os.path.basename(video_path)}")
        print(f"{'='*50}")
        
        # Skip already processed videos if resume option is enabled
        if args.resume:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video = os.path.join(output_folder, video_name, f'{video_name}_tracked.mp4')
            if os.path.exists(output_video):
                print(f"Video already processed (found {output_video}). Skipping...")
                skipped += 1
                continue
        
        try:
            if process_video(video_path):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {os.path.basename(video_path)}: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Processing complete: {successful} successful, {failed} failed, {skipped} skipped")
    if successful > 0:
        print(f"Output videos saved to: {output_folder}")
        print(f"Output data saved to: {output_folder_timeseries}")
    if failed > 0:
        print("Check the console output above for error details on failed videos")

if __name__ == "__main__":
    main()