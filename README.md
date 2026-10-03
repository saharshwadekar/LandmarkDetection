# Landmark Detection

An exploration of landmark recognition on the **[Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)** (GLDv2). The goal is to classify a photo by the landmark it shows. The notebooks cover data preparation, class-distribution analysis, and a VGG19 training pipeline with a batch generator that streams images from disk.

> More of my work at **[https://YOUR-PORTFOLIO-URL](https://YOUR-PORTFOLIO-URL)**

> **Status: work in progress.** The pipeline runs from start to finish, but no model has reached a useful accuracy yet. The class-distribution findings below explain why, and set out what comes next.

## Notebooks

| Notebook | Environment | Subset |
| --- | --- | --- |
| [`LandmarkDetection_SaharshWadekar.ipynb`](LandmarkDetection_SaharshWadekar.ipynb) | Google Colab + Drive | Image IDs starting with `00`: **16,157 images, 13,589 landmarks** |
| [`LandmarkDetectionv2.ipynb`](LandmarkDetectionv2.ipynb) | Local | IDs starting with `00`, `01` or `020`: **32,689 images, 24,576 landmarks** |

## Pipeline

1. **Load the index.** GLDv2's `train.csv` has 4,132,914 rows (`id`, `url`, `landmark_id`). The URL column is dropped.
2. **Take a subset by ID prefix.** The full dataset is about 500 GB, so the notebooks keep only IDs with a given prefix. GLDv2 shards images by the first three characters of their ID (`a/b/c/<id>.jpg`).
3. **Check files on disk.** Each row is checked against the image folder, and rows with missing files are dropped.
4. **Analyse the classes.** Images per landmark are counted and plotted.
5. **Encode labels.** `LabelEncoder` maps landmark IDs to class indices.
6. **Build the model.** A VGG19 architecture with BatchNormalization inserted before the convolutional blocks, and a new softmax head with one output per landmark.
7. **Train.** A custom `get_batch` generator reads, resizes (224 × 224) and normalises images 16 at a time, then calls `train_on_batch`. This keeps memory flat however large the subset is. The split is 80/20 train/validation.
8. **Evaluate.** Validation accuracy is computed per batch, and the most confident correct predictions are shown.

## What the data shows

This finding drove the next steps:

| Subset | Images | Landmarks | Mean images per landmark | Landmarks with ≤ 5 images |
| --- | --- | --- | --- | --- |
| v1 (`00`) | 16,157 | 13,589 | **1.19** | 13,549 (99.7%) |
| v2 (`00`, `01`, `020`) | 32,689 | 24,576 | 1.33 | 24,394 (99.3%) |

In the v1 subset, at least 75% of landmarks have **exactly one image** (the 75th percentile of the count is 1). A landmark with one image is in either the training set or the validation set, never both, so a classifier cannot learn it. Taking a slice by ID prefix spreads images thinly across almost every landmark, so the long tail of GLDv2 gets worse.

The model also has 195M–240M parameters, and most of them sit in the final dense layer (13,589 or 24,576 outputs). It is trained **from random weights** (`weights=None`) on about 13k images for one epoch, so it cannot converge.

## Next steps

- **Select by landmark, not by ID prefix.** Keep the top N landmarks with at least 20 images each, so every class appears in both training and validation.
- **Start from ImageNet weights** and fine-tune (EfficientNet or ResNet50), not train VGG19 from scratch.
- **Replace the classifier with retrieval.** Learn image embeddings (for example ArcFace loss) and match against a gallery. This is how the leading GLDv2 solutions handle the long tail.
- Report top-1 accuracy and **GAP** (Global Average Precision), the competition metric.
- Replace the Python batch loop with a `tf.data` pipeline, and use `class_weight`. The `weight_classes` flag is declared in the notebook but not used yet.

## Run it

1. Download `train.csv` and one or more image shards from the [GLDv2 repository](https://github.com/cvdfoundation/google-landmark).
2. Point `base_path` and the `train.csv` path at your copy. v1 expects Google Drive in Colab; v2 expects `./images/` and `./Train.csv`.
3. Install the dependencies: `pip install tensorflow keras opencv-python pandas numpy matplotlib pillow scikit-learn`
4. Run the cells in order.

## Tech

Python · TensorFlow / Keras · VGG19 · OpenCV · pandas · scikit-learn · Google Colab

## Author

**Saharsh Wadekar** · [Portfolio](https://YOUR-PORTFOLIO-URL) · [GitHub](https://github.com/saharshwadekar) · [LinkedIn](https://www.linkedin.com/in/saharshwadekar)
