# STL-10 Image Classification

## Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Layers**: 5 Conv layers with Batch Normalization, Max Pooling, GAP, Dropout (0.5), and 2 Fully Connected layers.
- **Input**: 96x96 RGB images.
  
Accuracy ~ 71%
Model Size ~ 150KB
## Training Procedure
- **Optimizer**: Adam (Cosine Annealing).
- **Technique**: Semi-Supervised Learning (Pseudo-labeling). The model uses high-confidence predictions (>90%) from the 100,000 unlabeled images to supplement the 5,000 labeled samples.
- **Data Augmentation**: Random Flips, Rotations, Crops, and Color Jittering. It also does cutmix and mixup augmentation

## Commands to Reproduce
1. **Install Dependencies**:
   `pip install -r requirements.txt`

2. **Run Training**:
   `python train.py --data ./data`

3. **Run Testing**:
   `python test.py --data ./data --model marvel_int8.pth`
