# Google Colab Setup Guide

This guide will help you run the Signature Forgery Detection project on Google Colab.

## Quick Start

1. **Open the Notebook**
   - Upload `Signature_Forgery_Detection_Colab.ipynb` to Google Colab
   - Or open directly from GitHub if the repository is public

2. **Run Setup Cells**
   - Execute cells in order (Shift+Enter)
   - The notebook will:
     - Install required packages
     - Clone repository from GitHub
     - Create project structure

3. **Upload Dataset**
   - Option A: Upload ZIP file containing `genuine/` and `forged/` folders
   - Option B: Mount Google Drive and copy data from Drive

4. **Train Model**
   - Run the training cell
   - Monitor training progress
   - View results and metrics

5. **Download Results**
   - Download trained models (.h5 files)
   - Download metrics and visualizations

## Features

- ✅ **Automatic Setup**: All dependencies installed automatically
- ✅ **GitHub Integration**: Clone repository directly
- ✅ **Flexible Data Upload**: ZIP file or Google Drive
- ✅ **Training with Contrastive Loss**: Optimized for Siamese networks
- ✅ **Visualization**: Training history and confusion matrix
- ✅ **Model Export**: Download trained models

## Requirements

- Google Colab account (free)
- Dataset with `genuine/` and `forged/` folders
- Internet connection (for GitHub clone)

## Dataset Structure

Your dataset should be organized as:
```
dataset.zip
├── genuine/
│   ├── signature1.png
│   ├── signature2.png
│   └── ...
└── forged/
    ├── fake1.png
    ├── fake2.png
    └── ...
```

## Training Parameters

Default parameters (can be modified in notebook):
- **Epochs**: 30
- **Batch Size**: 8
- **Learning Rate**: 0.001
- **Image Size**: 128x128
- **Loss Function**: Contrastive Loss (margin=1.0, threshold=0.5)

## Troubleshooting

### Issue: "Module not found"
**Solution**: Make sure you ran the GitHub clone cell and all source files are in `/content/src/`

### Issue: "Dataset not found"
**Solution**: 
- Check that you uploaded the dataset ZIP file
- Verify `data/genuine/` and `data/forged/` folders exist
- Run the verification cell to check structure

### Issue: "Out of memory"
**Solution**:
- Reduce batch size (e.g., from 8 to 4)
- Use smaller image size (e.g., 64x64 instead of 128x128)
- Restart runtime and clear memory

### Issue: "Training too slow"
**Solution**:
- Enable GPU runtime: Runtime → Change runtime type → GPU
- Increase batch size (if memory allows)
- Reduce number of epochs for testing

## GPU Usage

To use GPU acceleration:
1. Go to Runtime → Change runtime type
2. Select "GPU" as Hardware accelerator
3. Click Save
4. Restart runtime if needed

GPU significantly speeds up training (5-10x faster).

## Support

For issues or questions:
- Check the main README.md
- Review training logs in the notebook output
- Verify dataset structure matches requirements

