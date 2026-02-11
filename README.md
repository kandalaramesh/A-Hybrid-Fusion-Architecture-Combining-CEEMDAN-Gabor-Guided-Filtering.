```markdown
# LPAS MRI-PET Fusion

**Laplacian Pyramid Adaptive Selection** - Advanced MRI-PET fusion algorithm.

## Features
- CEEMDAN decomposition (MRI base/detail)
- NSST-like decomposition (PET Gabor-weighted)  
- Laplacian pyramid with adaptive LF ratios (55-65% MRI)
- VGG19 saliency-guided adaptation (layers 11-35)
- High-freq blending (60% MRI, 40% PET)
- YUV chroma neutralization + tone mapping
- Batch processing with per-image optimization

## Quick Setup
```bash
pip install torch torchvision opencv-python pillow numpy PyEMD
```

## Directory Structure
```
rData/
├── lasT1/      # T1 MRI (grayscale)
├── lasT2/      # T2 MRI (grayscale)
└── lasPET/     # PET (RGB)

rD2/folder-NSST-FusionsaliencyT15fixednewLs/
├── T1-PET/*.png
├── T2-PET/*.png
├── log_layers.csv
└── runtime.txt
```

## Run
```bash
python Proposed_LPAS-fusion.py
```
**Prompts**: `1 10` (images 1-10), Enter (all), `0` (skip)

## Pipeline
```
MRI ──CEEMDAN───► Base_m + Detail_m (60%)
PET ──NSST──────► Base_p + Detail_p (40%)
     ↓ Pyramid Fusion (55-65% MRI bias via VGG saliency)
     └─► Y_fused + UV_desaturated → RGB
```

## Sample Output
```csv
MRIType,Image,ChosenL,SaliencyCost,Time(sec)
T1-PET,image001,15,0.0234,2.847
T1-PET,image002,17,0.0198,2.912
```

## Key Parameters
| Component | Value | Purpose |
|-----------|-------|---------|
| Target Ratio | 0.55 | PET/MRI saliency balance |
| HF Blend | 60/40 | Structure preservation |
| LF Adaptive | 55-65% | Content-aware base fusion |

**VIT-AP University** | **Medical Imaging PhD Research** | **MIT License**
```

**Copy-paste this entire block into README.md**
