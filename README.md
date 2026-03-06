# YOLO-FPSA: YOLO + Frequency Phase-Shifted Attention

##  Overview
This repo modifies Ultralytics YOLO to add FFT-based frequency and phase-shifted attention for small object detection.

## Version Info
### ver2.0: 
- PhaseIFFT_1 block 추가(rgb channel input -> convert grayscale-> FFT -> phase IFFT)
- fpsa_1_1.yaml 

##  How to Use

```bash
conda env create -f environment.yml
conda activate FPSA_Paper

python ./scripts/train.py
