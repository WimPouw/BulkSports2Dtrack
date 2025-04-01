# Sports2D

A script to use Sports2D for bulk processing. See BodyMetaInfo for weight and mass for IK scaling.

# ToDo
- Timeseries should be seperated out into csv's
- Maybe visualizing forced vectors on video
- etc

## Installation Guide

### Prerequisites
- Python 3.10
- CUDA-compatible GPU, CUDA toolkit 11.8 is used here
- Conda

### Setup Environment

1. **Create and activate a conda environment**
   ```bash
   conda create -n Sports2D python=3.10 -y
   conda activate Sports2D
   ```

2. **Install OpenSim Python API**
   ```bash
   conda install -c opensim-org opensim -y
   ```

3. **Install Sports2D with Pose2Sim**
   ```bash
   pip install sports2d pose2sim
   ```

4. **Check CUDA compatibility**
   ```bash
   nvidia-smi
   ```

5. **Install PyTorch with CUDA support**
   *Note: Adjust the CUDA version based on your system*
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

6. **Install ONNX Runtime with GPU support**
   ```bash
   pip install onnxruntime-gpu
   ```

7. **Verify installation**
   ```bash
   python -c 'import torch; print(torch.cuda.is_available())'
   python -c 'import onnxruntime as ort; print(ort.get_available_providers())'
   ```

## Usage

Run the tracking script to do tracking and joint angles:
```bash
python track_sports2D.py --do_ik --use_detailed_model --mode performance --use_gpu
```

For simple version without ik scaling and cpu
```bash
python track_sports2d.py
```