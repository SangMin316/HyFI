# HyFI: Hyperbolic Feature Interpolation for Brain-Vision Alignment

## 📁 Repository Structure
```
HyFI/                           # Root directory
├── README.md
├── Analysis                   # Some analysis files
│   ├── check_the_retrieval.py # Retrieval results
│   └── plot_feature_dis.py    # Plot distribution of feature's distance from rooot 
├── base                       # Core implementation files
│   ├── data.py                # Data loading
│   ├── eeg_backbone.py        # EEG encoder backbone 
│   ├── inpating_data.py       # Inpainting data module
│   └── utils.py               # Utility functions including loss
│   └── hycoclip               # Inpainting data module
│       ├── checkpoints        # Check point for pre-train models
│       ├── encoders           
│       │   ├── image_encoders.py # Image encdoer for hycoclip
│       │   └── text_encoders.py  # Iext encdoer for hycoclip
│       ├── utils
│       │   ├── timer.py       
│       │   └── distributed.py 
│       ├── lorentz.py         # Lorentz manifold operations
│       ├── models.py          # MERU and HyCoCLIP models
│       └── tokenizer.py       # Tokenizer
├── configs
│   ├── MEG.yaml               # Configuration for MEG experiments
│   └── EEG.yaml               # Configuration for EEG experiments
├── exp                        # Directory for experiment results
├── preprocess
│   ├── process_eeg_whiten.py  # Script to preprocess and whiten EEG data
│   └── process_resize.py      # Script to resize image dataset
├── main.py                    # Main script for running experiments for HyFI
├── main_CLIP.py               # Main script for running experiments for CLIP interpolation
└── requirements.txt           # List of required Python packages

```


## Environment Setup
- Python 3.9
- Cuda 12.4
- PyTorch 2.6
- pytorch-lightning==2.5.1
- Required libraries are listed in `requirements.txt`.

```
pip install -r requirements.txt
```

## Running the Code
```
python main.py
```

## Acknowledgements

We would like to acknowledge the use of the following publicly available datasets:
- [A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758) [THINGS-EEG]
- [
THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior](https://pubmed.ncbi.nlm.nih.gov/36847339/) [THINGS-MEG]

This codebase is inspired by several previous works in neural decoding:
- [Decoding Natural Images from EEG for Object Recognition](https://github.com/eeyhsong/NICE-EEG) [ICLR 2024]
- [Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://github.com/dongyangli-del/EEG_Image_decode) [NeurIPS 2024]
- [Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](https://github.com/HaitaoWuTJU/Uncertainty-aware-Blur-Prior) [CVPR 2025]

This codebase is inspired by several previous works in hyperbolic representation learning:
- [hyperbolic image text representations](https://github.com/facebookresearch/meru) [ICML 2023]
- [Compositional entailment learning for hyperbolic vision-language models](https://github.com/PalAvik/hycoclip) [ICLR 2025]



