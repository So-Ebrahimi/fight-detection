# Fight Detection

A deep learning model for detecting fights in video sequences using 3D CNN.

## Usage

### Training

Train the model using train and validation sets:
```bash
python train.py
```

The training script will:
- Load data from `dataset/train` and `dataset/valid`
- Train for 30 epochs with validation monitoring
- Save the best model to `fight_model.pth` based on validation loss

### Testing

Test the model on the test set:
```bash
python test_all_videos.py
```

This will evaluate the model on `dataset/test` and provide:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Per-class accuracy
- Detailed results for each video

### Single Video Testing

Test a single video:
```bash
python test_video.py
```

## Model Architecture

The model uses a 3D CNN architecture:
- 3D Convolutional layers for spatiotemporal feature extraction
- Batch normalization and max pooling
- Global average pooling
- Fully connected classifier

## Files

- `train.py` - Training script with train/validation split
- `test_all_videos.py` - Evaluation on test set
- `test_video.py` - Single video testing
- `model.py` - Model architecture
- `dataset.py` - Dataset loader
- `split_dataset.py` - Script to split validation into test/validate
