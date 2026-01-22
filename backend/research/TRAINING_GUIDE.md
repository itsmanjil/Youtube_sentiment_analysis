# Training Guide: Hybrid CNN-BiLSTM-Attention Model

Complete guide for training the hybrid deep learning sentiment analysis model for your Master's thesis.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Training Options](#training-options)
4. [Configuration Files](#configuration-files)
5. [Monitoring Training](#monitoring-training)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Quick Start

### Prerequisites

```bash
# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard scikit-learn pandas numpy pyyaml

# Download GloVe embeddings (optional but recommended)
# Visit: https://nlp.stanford.edu/projects/glove/
# Download glove.6B.zip (822MB) and extract
```

### Basic Training

```bash
# Navigate to research directory
cd backend/research

# Train with config file (recommended)
python train_hybrid_dl.py --config config/hybrid_dl_config.yaml

# Train with minimal CLI arguments
python train_hybrid_dl.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --test_csv data/test.csv \
    --max_epochs 50 \
    --batch_size 32
```

### Expected Output

```
================================================================================
EXPERIMENT: hybrid_dl_20260114_153045
================================================================================
Output directory: output/hybrid_dl_20260114_153045
TensorBoard logs: ./runs
================================================================================

✅ Configuration saved to: output/hybrid_dl_20260114_153045/config.yaml

================================================================================
STEP 1: LOADING DATA
================================================================================

Loading Data from CSV Files
Building vocabulary from data/train.csv...
✅ Built vocabulary with 15847 tokens
✅ Saved vocabulary to output/hybrid_dl_20260114_153045/vocab.pkl
✅ Loaded 8000 samples from data/train.csv
   Label distribution: {0: 2667, 1: 2666, 2: 2667}
✅ Loaded 1000 samples from data/val.csv
   Label distribution: {0: 333, 1: 334, 2: 333}

✅ DataLoaders created:
   Train batches: 250
   Val batches:   32

================================================================================
STEP 2: BUILDING MODEL
================================================================================

✅ Model created
   Total parameters: 2,487,907
   Trainable parameters: 2,487,907

Loading GloVe embeddings from: embeddings/glove.6B.300d.txt
✅ GloVe embeddings loaded
   Vocabulary coverage: 87.34%

================================================================================
STEP 3: SETTING UP TRAINING
================================================================================

✅ Optimizer: Adam (lr=0.001, weight_decay=0.0001)
✅ LR Scheduler: ReduceLROnPlateau (patience=3)
✅ Early Stopping (patience=7)
✅ Model Checkpoint (saving best model)

================================================================================
STEP 4: TRAINING
================================================================================
📊 TensorBoard logging to: runs/hybrid_dl_20260114_153045
   Run: tensorboard --logdir=./runs

Device: cuda
Model parameters: 2,487,907
Trainable parameters: 2,487,907
Training samples: 8000
Validation samples: 1000
Batch size: 32
Batches per epoch: 250
Max epochs: 50
================================================================================

================================================================================
Epoch 1/50
================================================================================
  Batch 10/250 (4.0%) - Loss: 0.9834
  Batch 20/250 (8.0%) - Loss: 0.8921
  ...

📊 Epoch 1 Summary (Time: 45.23s):
   train_accuracy: 0.6234
   train_f1_macro: 0.6101
   train_avg_loss: 0.8456
   val_accuracy: 0.6540
   val_f1_macro: 0.6423
   val_avg_loss: 0.7821
   val_cohen_kappa: 0.4813

✅ val_f1_macro increased to 0.6423
💾 [BEST] Checkpoint saved: output/.../checkpoints/hybrid_epoch01_f10.6423.pt
```

---

## Data Preparation

### CSV Format

Your training data must be in CSV format with text and label columns:

```csv
text,label
"This movie is amazing! Best film ever!",positive
"Terrible experience. Would not recommend.",negative
"It's okay, nothing special.",neutral
```

### Label Format

The model supports both string and integer labels:

**String labels** (case-insensitive):
- `'positive'`, `'Positive'`, or `'POSITIVE'` → class 2
- `'neutral'`, `'Neutral'`, or `'NEUTRAL'` → class 1
- `'negative'`, `'Negative'`, or `'NEGATIVE'` → class 0

**Integer labels**:
- `0` → Negative
- `1` → Neutral
- `2` → Positive

### Data Split Recommendations

For thesis-level experiments:

1. **Standard Split** (if you have enough data):
   - Training: 70% (~7,000+ samples)
   - Validation: 15% (~1,500+ samples)
   - Test: 15% (~1,500+ samples)

2. **10-Fold Cross-Validation** (for smaller datasets):
   ```python
   # Use create_k_fold_loaders from data_loaders.py
   from data.data_loaders import create_k_fold_loaders

   for fold in range(10):
       train_loader, val_loader = create_k_fold_loaders(
           dataset, n_folds=10, fold_index=fold
       )
       # Train model for this fold
   ```

### Preparing YouTube Comments

If using the existing Django backend:

```python
# Export YouTube comments to CSV
import pandas as pd
from app.models import YouTubeComment

# Get labeled comments
comments = YouTubeComment.objects.filter(sentiment__isnull=False)

data = []
for comment in comments:
    data.append({
        'text': comment.text,
        'label': comment.sentiment  # 'positive', 'negative', or 'neutral'
    })

df = pd.DataFrame(data)

# Stratified split
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

# Save to CSV
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
```

---

## Training Options

### Command-Line Arguments

#### Essential Arguments

```bash
--config PATH              # Path to YAML config file (recommended)
--train_csv PATH          # Training data CSV
--val_csv PATH            # Validation data CSV
--test_csv PATH           # Test data CSV (optional)
```

#### Model Architecture

```bash
--embedding_dim INT       # Embedding dimension (default: 300)
--max_len INT             # Maximum sequence length (default: 200)
--num_classes INT         # Number of classes (default: 3)
```

#### Training Hyperparameters

```bash
--batch_size INT          # Batch size (default: 32)
--max_epochs INT          # Maximum epochs (default: 50)
--learning_rate FLOAT     # Learning rate (default: 0.001)
--weight_decay FLOAT      # L2 regularization (default: 0.0001)
--gradient_clip FLOAT     # Gradient clipping norm (default: 5.0)
```

#### Early Stopping

```bash
--early_stopping_patience INT      # Patience epochs (default: 7)
--early_stopping_monitor METRIC    # Metric to monitor (default: val_f1_macro)
```

#### Vocabulary

```bash
--vocab_path PATH         # Load existing vocabulary
--vocab_max_size INT      # Max vocab size (default: 20000)
--vocab_min_freq INT      # Min word frequency (default: 2)
```

#### Embeddings

```bash
--glove_path PATH         # Path to GloVe embeddings
```

#### Paths

```bash
--output_dir PATH         # Output directory (default: ./output)
--tensorboard_dir PATH    # TensorBoard logs (default: ./runs)
--experiment_name NAME    # Experiment name (default: timestamp)
```

#### Device

```bash
--device auto|cuda|cpu|mps   # Training device (default: auto)
```

#### Resume Training

```bash
--resume PATH             # Resume from checkpoint
```

#### Other

```bash
--seed INT                # Random seed (default: 42)
--num_workers INT         # Data loading workers (default: 0)
```

---

## Configuration Files

Using YAML config files is **recommended** for reproducibility and cleaner experimentation.

### Example: `config/hybrid_dl_config.yaml`

```yaml
# Model Architecture
model:
  embedding_dim: 300
  vocab_size: 20000
  max_len: 200
  num_classes: 3

  cnn:
    filter_sizes: [3, 4, 5]
    num_filters: 128
    dropout: 0.3

  bilstm:
    hidden_size: 128
    num_layers: 2
    dropout: 0.3

  attention:
    num_heads: 4
    dropout: 0.1

  classifier:
    hidden_sizes: [256, 128]
    dropout: [0.5, 0.4]

# Training
training:
  batch_size: 32
  max_epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  gradient_clip: 5.0

  lr_scheduler:
    type: 'reduce_on_plateau'  # or 'step', 'cosine'
    patience: 3
    factor: 0.5

  early_stopping:
    enabled: true
    patience: 7
    monitor: 'val_f1_macro'

# Data
data:
  max_len: 200
  vocab_max_size: 20000
  vocab_min_freq: 2
  text_column: 'text'
  label_column: 'label'

# Embeddings
embeddings:
  type: 'glove'
  path: './embeddings/glove.6B.300d.txt'
  dim: 300
  trainable: true

# Paths
train_csv: 'data/train.csv'
val_csv: 'data/val.csv'
test_csv: 'data/test.csv'
output_dir: './output'
tensorboard_dir: './runs'

# Other
device: 'auto'
seed: 42
num_workers: 0
```

### Override Config with CLI

CLI arguments take precedence over config file:

```bash
# Use config but override batch size and learning rate
python train_hybrid_dl.py \
    --config config/hybrid_dl_config.yaml \
    --batch_size 64 \
    --learning_rate 0.0005
```

---

## Monitoring Training

### TensorBoard

TensorBoard provides real-time visualization of training progress:

```bash
# Start TensorBoard (in a separate terminal)
tensorboard --logdir=./runs

# Open browser to http://localhost:6006
```

**Available Visualizations:**

1. **Scalars**:
   - Training loss, accuracy, F1-macro, precision, recall
   - Validation loss, accuracy, F1-macro, precision, recall
   - Per-class F1 scores
   - Learning rate
   - Cohen's Kappa

2. **Comparison Plots**:
   - Train vs Validation Loss
   - Train vs Validation F1-Macro

3. **Distributions** (if enabled):
   - Model weights
   - Gradient norms

### Training History

The training history is automatically saved to `output/<experiment>/training_history.json`:

```json
{
  "history": {
    "train_loss": [0.8456, 0.7234, 0.6543, ...],
    "train_f1_macro": [0.6101, 0.6789, 0.7234, ...],
    "val_loss": [0.7821, 0.6987, 0.6234, ...],
    "val_f1_macro": [0.6423, 0.7012, 0.7456, ...]
  },
  "metadata": {
    "saved_at": "2026-01-14T15:45:32",
    "total_epochs": 23
  }
}
```

### Plot Training Curves

```python
import json
import matplotlib.pyplot as plt

# Load history
with open('output/hybrid_dl_20260114_153045/training_history.json') as f:
    data = json.load(f)

history = data['history']

# Plot F1-Macro
plt.figure(figsize=(10, 6))
plt.plot(history['train_f1_macro'], label='Train')
plt.plot(history['val_f1_macro'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('F1-Macro')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_curves.png', dpi=300)
plt.show()
```

---

## Best Practices

### 1. Start with Baseline

Before heavy tuning, establish a baseline:

```bash
# Quick baseline run (10 epochs, no fancy stuff)
python train_hybrid_dl.py \
    --config config/hybrid_dl_config.yaml \
    --max_epochs 10 \
    --experiment_name baseline
```

Expected baseline performance:
- **F1-Macro**: 0.75-0.82
- **Accuracy**: 0.76-0.84

### 2. Hyperparameter Tuning Order

Tune hyperparameters in this order:

1. **Learning rate** (most important)
   - Try: [0.0001, 0.0005, 0.001, 0.002, 0.005]
   - Use learning rate finder if available

2. **Batch size**
   - Larger = faster, more stable (if GPU memory allows)
   - Try: [16, 32, 64, 128]

3. **Architecture**
   - CNN filters: [64, 96, 128, 160]
   - LSTM hidden: [64, 96, 128, 192, 256]

4. **Regularization**
   - Dropout: [0.2, 0.3, 0.4, 0.5]
   - Weight decay: [0.00001, 0.0001, 0.001]

5. **Sequence length**
   - Shorter = faster training
   - Analyze your data's length distribution first

### 3. Monitor Overfitting

**Signs of overfitting:**
- Train F1 >> Val F1 (gap > 0.05)
- Train loss decreases, val loss increases

**Solutions:**
1. Increase dropout rates
2. Add weight decay
3. Reduce model size
4. Get more training data
5. Use early stopping (already enabled)

### 4. GPU Memory Management

If you encounter OOM (Out of Memory) errors:

```bash
# Reduce batch size
--batch_size 16

# Reduce sequence length
--max_len 150

# Reduce model size
# Edit config.yaml:
#   cnn.num_filters: 96 (instead of 128)
#   bilstm.hidden_size: 96 (instead of 128)
```

### 5. Training Time Estimates

On a typical setup (NVIDIA RTX 3080, batch_size=32):
- **Per epoch**: 30-60 seconds (8K samples)
- **Full training** (50 epochs with early stopping): 30-60 minutes
- **Expected stopping**: Usually 20-30 epochs

### 6. Reproducibility

For reproducible results:

```bash
# Always use the same seed
--seed 42

# Disable CUDA non-deterministic operations
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

---

## Troubleshooting

### Problem: Import Errors

```
ModuleNotFoundError: No module named 'architectures'
```

**Solution**: Make sure you're in the correct directory:
```bash
cd backend/research
python train_hybrid_dl.py --config config/hybrid_dl_config.yaml
```

### Problem: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size: `--batch_size 16`
2. Reduce sequence length: `--max_len 150`
3. Reduce model size (edit config)
4. Clear GPU cache: `torch.cuda.empty_cache()`

### Problem: Vocabulary Coverage Low

```
✅ GloVe embeddings loaded
   Vocabulary coverage: 45.23%
```

**Solutions**:
1. Use a larger GloVe file (e.g., `glove.840B.300d.txt`)
2. Train embeddings from scratch (set `embeddings.trainable: true`)
3. Check if your text is properly preprocessed

### Problem: Training Very Slow

**Causes and Solutions**:

1. **CPU training instead of GPU**:
   ```bash
   # Check if CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"

   # Force GPU
   --device cuda
   ```

2. **Too many data loading workers on Windows**:
   ```bash
   # Windows doesn't handle multiprocessing well
   --num_workers 0
   ```

3. **Large vocabulary**:
   ```bash
   # Reduce vocabulary size
   --vocab_max_size 10000
   ```

### Problem: Model Not Learning (Loss Not Decreasing)

**Diagnostic Steps**:

1. **Check data labels**:
   ```python
   import pandas as pd
   df = pd.read_csv('data/train.csv')
   print(df['label'].value_counts())  # Should be roughly balanced
   ```

2. **Check for label mapping issues**:
   ```python
   # Make sure labels are correct
   unique_labels = df['label'].unique()
   print(unique_labels)  # Should be in ['positive', 'negative', 'neutral'] or [0, 1, 2]
   ```

3. **Try higher learning rate**:
   ```bash
   --learning_rate 0.002
   ```

4. **Check for data preprocessing issues**:
   - Ensure text is not empty
   - Check for HTML tags or special characters
   - Verify encoding (should be UTF-8)

### Problem: Val F1 Stuck at ~0.33 (Random Baseline)

This means the model is predicting everything as one class.

**Solutions**:
1. **Check class balance** in your data
2. **Use class weights**:
   ```python
   # In train_hybrid_dl.py, modify criterion:
   from sklearn.utils.class_weight import compute_class_weight

   class_weights = compute_class_weight(
       'balanced',
       classes=np.array([0, 1, 2]),
       y=train_labels
   )
   criterion = nn.CrossEntropyLoss(
       weight=torch.FloatTensor(class_weights)
   )
   ```

3. **Lower learning rate**: Model might be diverging

---

## Advanced Usage

### 1. K-Fold Cross-Validation

For thesis experiments requiring k-fold CV:

```python
# create_kfold_experiment.py
from data.data_loaders import create_k_fold_loaders
from data.dataset import CSVSentimentDataset
from data.preprocessing import Vocabulary
import pandas as pd

# Load all data
df = pd.read_csv('data/all_data.csv')
texts = df['text'].tolist()
labels = df['label'].tolist()

# Build vocab
vocab = Vocabulary(max_size=20000, min_freq=2)
vocab.build_from_texts(texts)

# Create dataset
from data.dataset import SentimentDataset
dataset = SentimentDataset(texts, labels, vocab)

# Train on each fold
results = []
for fold in range(10):
    print(f"\n{'='*80}")
    print(f"TRAINING FOLD {fold + 1}/10")
    print(f"{'='*80}\n")

    train_loader, val_loader = create_k_fold_loaders(
        dataset, n_folds=10, fold_index=fold
    )

    # Train model (create new model for each fold)
    model = HybridCNNBiLSTM(...)
    trainer = HybridDLTrainer(...)
    trainer.train()

    # Evaluate
    fold_metrics = trainer.evaluate_on_test(val_loader)
    results.append(fold_metrics)

# Aggregate results
import numpy as np
print("\n10-FOLD CROSS-VALIDATION RESULTS:")
print(f"F1-Macro: {np.mean([r['f1_macro'] for r in results]):.4f} "
      f"± {np.std([r['f1_macro'] for r in results]):.4f}")
```

### 2. Ensemble with Existing Models

Combine the hybrid DL model with existing LogReg, SVM, TF-IDF:

```python
# After training hybrid model
from src.sentiment import get_sentiment_engine

# Load trained hybrid model
hybrid_engine = get_sentiment_engine('hybrid_dl', model_path='output/.../final_model.pt')

# Create weighted ensemble
ensemble = get_sentiment_engine(
    'ensemble',
    base_models=['logreg', 'svm', 'tfidf', 'hybrid_dl'],
    weights={'logreg': 0.15, 'svm': 0.25, 'tfidf': 0.20, 'hybrid_dl': 0.40},
)

# Use in production
result = ensemble.analyze("This movie is amazing!")
print(result)
```

### 3. Transfer Learning from Checkpoint

Fine-tune a pretrained model on new data:

```python
# Load pretrained model
checkpoint = torch.load('pretrained_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Freeze early layers
for name, param in model.named_parameters():
    if 'embedding' in name or 'cnn' in name:
        param.requires_grad = False

# Train only the classifier
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001  # Lower LR for fine-tuning
)
```

### 4. Attention Visualization

Visualize what the model is paying attention to:

```python
# Extract attention weights during inference
model.eval()
with torch.no_grad():
    logits, attention_weights = model(input_ids)

# attention_weights shape: [batch_size, num_heads, seq_len, seq_len]

# Visualize for first sample
import matplotlib.pyplot as plt
import seaborn as sns

att = attention_weights[0, 0].cpu().numpy()  # First sample, first head

plt.figure(figsize=(10, 8))
sns.heatmap(att, cmap='viridis', cbar=True)
plt.title('Self-Attention Weights (Head 1)')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.savefig('attention_visualization.png')
```

---

## Integration with Thesis Evaluation Framework

To use the trained model with the existing `evaluation_framework.py`:

```python
from research.evaluation_framework import ThesisEvaluationFramework
from app.deep_models import HybridDLSentimentEngine
import numpy as np

# Load trained model
model_engine = HybridDLSentimentEngine(
    model_path='output/hybrid_dl_20260114_153045/final_model.pt',
    vocab_path='output/hybrid_dl_20260114_153045/vocab.pkl'
)

# Prepare data
X_texts = ["text1", "text2", ...]  # Your text data
y_true = np.array([0, 1, 2, ...])  # True labels

# Create model function for evaluation framework
def model_fn():
    class ModelWrapper:
        def fit(self, X, y):
            pass  # Already trained

        def predict(self, X):
            predictions = []
            for text in X:
                result = model_engine.analyze(text)
                # Map label back to int
                label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
                predictions.append(label_map[result.label])
            return np.array(predictions)

    return ModelWrapper()

# Evaluate with 10-fold CV metrics
evaluator = ThesisEvaluationFramework(n_folds=10, random_state=42)

# Note: This will retrain on folds, so use for baseline comparison
# For your hybrid model, run separate inference and compute metrics
```

---

## Questions?

If you encounter issues not covered here:

1. Check the code comments in the implementation files
2. Review error messages carefully - they often contain the solution
3. Verify your Python environment and package versions
4. Ensure your data format matches the expected CSV structure

**Happy Training! 🚀**
