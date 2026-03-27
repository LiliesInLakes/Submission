# STL-10 Image Classification

## Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Layers**: 3 Conv layers with Batch Normalization, Max Pooling, Dropout (0.5), and 2 Fully Connected layers.
- **Input**: 96x96 RGB images.

## Training Procedure
- **Optimizer**: Adam (Learning Rate: 0.001).
- **Technique**: Semi-Supervised Learning (Pseudo-labeling). The model uses high-confidence predictions (>95%) from the 100,000 unlabeled images to supplement the 5,000 labeled samples.
- **Data Augmentation**: Random Flips, Rotations, Crops, and Color Jittering.

## Commands to Reproduce
1. **Install Dependencies**:
   `pip install -r requirements.txt`

2. **Run Training**:
   `python train.py --data ./data --epochs 50 --batch_size 64`

3. **Run Testing**:
   `python test.py --data ./data`
