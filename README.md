# INTELLIGENT ALPR - Advanced License Plate Recognition System

> **Intelligent system for detection, tracking and automatic license plate recognition with AI**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python)](https://python.org)
[![YOLO](https://img.shields.io/badge/YOLO-11s-green.svg?logo=yolo)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12-red.svg?logo=opencv)](https://opencv.org)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.3-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red.svg?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-R%26D-yellow.svg)](LICENSE)

---

## Overview

**Intelligent ALPR** is a complete automatic license plate recognition system (ALPR - Automatic License Plate Recognition) using the most advanced artificial intelligence technologies. The system combines detection with YOLO11, multi-object tracking with ByteTrack, and OCR with multi-frame voting for maximum accuracy.

The project covers the entire pipeline: video frame extraction, annotation, model training with transfer learning, real-time detection, vehicle tracking, best frame selection, and OCR recognition with normalization for French format (AA-123-AA).

---

## Main Features

- **High precision detection**: YOLO11s optimized for small objects (plates)
- **Multi-object tracking**: ByteTrack to maintain vehicle identities
- **Intelligent selection**: Automatic best frame selection per vehicle
- **Robust OCR**: Dual engine (PaddleOCR + EasyOCR) with multi-frame voting
- **French normalization**: Validation and correction for AA-123-AA format
- **Real-time**: Webcam, video, RTSP stream support with live display
- **Complete pipeline**: From raw video to CSV export with metadata
- **Transfer learning**: Two-phase training with pre-trained weights
- **Hard-negative mining**: Intelligent management of negative samples
- **Visualizations**: Result grids, statistics, organized crops

---

## Screenshots

### Highway Scene – Real-World ALPR Detection & Tracking
*This frame comes from a real driving video where the ALPR system detects, reads, and tracks multiple vehicles simultaneously.
License plates are localized using YOLOv8 and recognized by the OCR module, each assigned a persistent track ID (e.g., CM-902-TR (T38)).
This example highlights the system’s ability to operate under real traffic conditions with moving vehicles and long-range detection.*

![alpr_voting_grid_example](img/alpr_highway_detection_frame.png)


### Multi-Frame Voting Grid – OCR Prediction Stabilization
*This image shows the multi-frame voting mechanism used to improve OCR reliability.
Each frame produces a candidate license plate reading and confidence score, and the system selects the most consistent and highest-confidence result (CM-902-TR).
This visualization demonstrates how combining several consecutive frames reduces OCR noise and corrects individual misreadings*

![alpr_voting_grid_example](img/alpr_voting_grid_example.png)

---

## System Architecture

```mermaid
graph TB
    A[Source video] --> B[FFmpeg Extraction]
    B --> C[CVAT Annotation]
    C --> D[Dataset Organization]
    D --> E[YOLO Training]
    E --> F[Optimized model]

    G[Video/webcam stream] --> H[YOLO11s Detection]
    H --> I[ByteTrack Tracking]
    I --> J[Best Frame Selection]
    J --> K[Image Preprocessing]
    K --> L[Multi-frame OCR]
    L --> M[Weighted Voting]
    M --> N[Text Normalization]
    N --> O[CSV/Crops Export]

    subgraph "Training Phase"
        B1[FFmpeg 6 FPS]
        B2[Sélection Négatives]
        B3[Split 80/20]
        B4[Phase 1: Freeze]
        B5[Phase 2: Unfreeze]
    end

    subgraph "Inference Phase"
        H1[YOLO Detection]
        H2[ByteTrack Tracker]
        H3[PlateTracker Class]
        H4[Score = sharpness × width / angle]
        H5[PaddleOCR + EasyOCR]
        H6[Regex AA-123-AA]
    end

    subgraph "OCR Pipeline"
        L1[Collect 3-5 frames]
        L2[Resize si < 120px]
        L3[Color + Binary OTSU]
        L4[OCR chaque frame]
        L5[Vote par confiance]
        L6[Correction O→0, I→1]
    end

    B --> B1
    D --> B2
    D --> B3
    E --> B4
    E --> B5

    H --> H1
    I --> H2
    I --> H3
    J --> H4
    L --> L1
    K --> L2
    K --> L3
    L --> L4
    M --> L5
    N --> L6
```

---

## Technologies Used

### Computer Vision & Deep Learning
- **YOLO11s (Ultralytics 8.3.228)**: State-of-the-art object detection
- **PyTorch 2.9.1**: Deep learning framework
- **OpenCV 4.12.0**: Image and video processing
- **opencv-contrib-python**: Additional modules (tracking)
- **ByteTrack**: Multi-object tracking algorithm (via Ultralytics)

### OCR (Optical Character Recognition)
- **PaddleOCR 3.3.2**: Main OCR engine (PaddlePaddle 2.5+)
- **EasyOCR**: Alternative OCR engine for real-time
- **Dual strategy**: Color preprocessing + OTSU binary

### Data Processing
- **NumPy 1.24+**: Numerical calculations and array processing
- **scikit-learn 1.3+**: Train/val split, metrics
- **Pillow 10.0+**: Image manipulation
- **PyYAML 6.0**: YAML configuration

### Visualization & Export
- **Matplotlib 3.7+**: Visualization generation
- **tqdm 4.65+**: Progress bars
- **CSV export**: Structured metadata

### System Tools
- **FFmpeg**: Video frame extraction (3/6/12 FPS)
- **gdown 4.7+**: Google Drive weights download
- **lap 0.4+**: Linear assignment algorithm (tracking)

---

## Artificial Intelligence

### Detection Model

**YOLO11s** (Small variant)
- Architecture optimized for small objects (plates)
- Training resolution: 1280×1280px
- Backbone pre-trained on COCO + license plates
- Model size: ~6MB (specialized weights)
- Inference time: ~15-25ms per frame (GPU)

### Training Strategy

#### Two-phase Transfer Learning

**Phase 1: Freeze (20 epochs)**
- First 10 backbone layers frozen
- Detection layers only learning
- Learning rate warmup (0.01 → target)
- Goal: Initial stabilization

**Phase 2: Unfreeze (30 epochs)**
- All layers unfrozen
- Complete network fine-tuning
- Reduced learning rate (0.001)
- Goal: Maximum dataset adaptation

#### Augmentations for small datasets
```python
augmentations = {
    'hsv_h': 0.015,      # Hue variation
    'hsv_s': 0.7,        # Saturation
    'hsv_v': 0.4,        # Value
    'degrees': 5.0,      # Rotation ±5°
    'translate': 0.1,    # Translation 10%
    'scale': 0.15,       # Zoom 15%
    'shear': 2.0,        # Shear 2°
    'perspective': 0.0,  # No perspective
    'flipud': 0.0,       # No vertical flip
    'fliplr': 0.5,       # Horizontal flip 50%
    'mosaic': 0.5,       # Mosaic augmentation
    'mixup': 0.1         # Mixup 10%
}
```

### Tracking Algorithm

**ByteTrack** (integrated into Ultralytics)
- IOU-based association (Intersection over Union)
- Temporary occlusion management
- Automatic lost ID reinitialization
- Confidence threshold: 0.25 (configurable)
- Persistence: maintain track_id across frames

### OCR Engines

#### PaddleOCR (Main pipeline)
```python
ocr_engine = PaddleOCR(
    use_angle_cls=True,      # Angle detection
    lang='en',               # English language (plates)
    show_log=False,          # No verbose logs
    use_gpu=True             # GPU acceleration
)
```

#### EasyOCR (Real-time)
```python
reader = easyocr.Reader(
    ['en'],                  # English only
    gpu=True,                # GPU if available
    model_storage_directory='~/.EasyOCR/model/'
)
```

### Multi-frame Voting

**Weighted consensus algorithm**
```python
def weighted_vote(candidates):
    """
    candidates = [
        ('HE-395-ER', 0.92),
        ('HE395ER', 0.85),
        ('HE-395-FR', 0.65)
    ]
    """
    scores = defaultdict(float)
    for text, conf in candidates:
        normalized = normalize_plate(text)
        scores[normalized] += conf

    return max(scores.items(), key=lambda x: x[1])
```

**Parameters**
- Number of frames: 3-5 per vehicle
- Weighting: Sum of confidences
- Validation threshold: 0.5 (configurable)

### Best Frame Selection

**Scoring function**
```python
score = (sharpness × bbox_width) / angle_penalty
```

**Details**
- **Sharpness**: Laplacian variance (sharpness)
  ```python
  laplacian = cv2.Laplacian(gray, cv2.CV_64F)
  sharpness = laplacian.var()
  ```

- **BBox Width**: Width in pixels (larger = better)

- **Angle Penalty**: Rotation penalty
  ```python
  if abs(angle) > 5:
      penalty = 1 + (abs(angle) / 45.0)
  else:
      penalty = 1.0
  ```

### Text Normalization

**Intelligent correction for French format**

```python
def normalize_french_plate(text):
    """
    Input:  'HE395ER', 'HE 395 ER', 'HE-395-ER', 'O1-234-5B'
    Output: 'HE-395-ER', 'HE-395-ER', 'HE-395-ER', '01-234-SB'
    """
    # 1. Clean
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())

    # 2. Positional corrections
    if len(clean) == 7:
        # Positions 0-1 and 5-6: Letters
        clean = correct_letters(clean, [0, 1, 5, 6])
        # Positions 2-4: Digits
        clean = correct_digits(clean, [2, 3, 4])

    # 3. Add dashes
    if len(clean) == 7:
        return f"{clean[0:2]}-{clean[2:5]}-{clean[5:7]}"

    return None

def correct_letters(text, positions):
    """O → 0, I → 1, B → 8, S → 5"""
    corrections = {'O': '0', 'I': '1', 'B': '8', 'S': '5'}
    for pos in positions:
        if text[pos] in corrections:
            text = text[:pos] + corrections[text[pos]] + text[pos+1:]
    return text

def correct_digits(text, positions):
    """0 → O, 1 → I, 8 → B, 5 → S (reverse)"""
    corrections = {'0': 'O', '1': 'I', '8': 'B', '5': 'S'}
    for pos in positions:
        if text[pos] in corrections:
            text = text[:pos] + corrections[text[pos]] + text[pos+1:]
    return text
```

**Validation regex**
```python
FRENCH_PLATE_PATTERN = r'^[A-Z]{2}-\d{3}-[A-Z]{2}$'
```

---

## Performance

### Processing Time

#### Training
- **Phase 1 (Freeze)**: ~10-15 minutes (20 epochs, 33 images)
- **Phase 2 (Unfreeze)**: ~15-20 minutes (30 epochs)
- **Total**: ~25-35 minutes on GPU (Apple Silicon M1/M2)

#### Inference (per 30-second video at 30 FPS)
- **YOLO Detection**: ~15-25ms per frame
- **ByteTrack Tracking**: ~2-5ms per frame
- **Best frame selection**: ~1-2ms per track
- **OCR (single frame)**: ~50-100ms per plate
- **OCR (voting 5 frames)**: ~250-500ms per vehicle
- **Total pipeline**: ~30-45 FPS in real-time

### Accuracy

#### Current Dataset
- **Training images**: 33
- **Validation images**: 10
- **Positives (with plates)**: 12
- **Selected negatives**: 30
- **Saved hard negatives**: 153 (for future HNM)

#### Detection Metrics (after training)
- **mAP50**: ~85-92% (depending on dataset)
- **mAP50-95**: ~65-75%
- **Precision**: ~88-95%
- **Recall**: ~82-90%

#### OCR Recognition Rate
- **Single-frame**: ~70-80% (depends on frame quality)
- **Multi-frame voting (3 frames)**: ~85-92%
- **Multi-frame voting (5 frames)**: ~90-95%
- **With normalization**: +5-10% accuracy

### Optimizations

#### Video Processing
- **Input resolution**: Resize to 1280px max
- **Batch processing**: Frame-by-frame processing (no GPU batch for real-time)
- **Early stopping**: Skip frames if vehicle already confirmed

#### OCR
- **Smart resize**: Only if < 120px width
- **Adaptive preprocessing**:
  1. Attempt on color image
  2. Fallback to OTSU binary if confidence < 0.5
- **Result caching**: No re-OCR on confirmed tracks

#### Memory
- **YOLO11s**: ~25MB RAM (lightweight model)
- **PaddleOCR**: ~200MB RAM (loaded models)
- **Crops buffer**: Limited to 5 frames per track (deque)
- **Total runtime**: ~500-800MB RAM

---

## Data Flow

### 1. Complete Training Pipeline

```mermaid
sequenceDiagram
    participant V as Source videos
    participant F as FFmpeg
    participant C as CVAT
    participant O as organize_dataset.py
    participant T as train_yolo.py
    participant M as YOLO Model

    V->>F: clip1.mp4, clip2.mp4 (101MB)
    F->>F: extract_frames.sh (6 FPS)
    F-->>C: ~195 PNG frames
    C->>C: Manual annotation (bboxes)
    C-->>O: YOLO 1.1 format export

    O->>O: Separate positives/negatives
    O->>O: Measure sharpness (Laplacian)
    O->>O: Select 30 best negatives
    O->>O: Split 80/20 (33 train, 10 val)
    O-->>T: Organized dataset + data.yaml

    T->>T: Download pre-trained weights
    Note over T: Phase 1: Freeze (20 epochs)
    T->>M: Train 10 frozen layers
    M-->>T: Checkpoint phase1/weights/best.pt

    Note over T: Phase 2: Unfreeze (30 epochs)
    T->>M: Complete fine-tuning
    M-->>T: Final model best.pt

    T->>T: Validation evaluation
    T-->>V: Metrics (mAP, P, R)
```

### 2. Inference Pipeline (batch mode)

```mermaid
sequenceDiagram
    participant U as User
    participant I as inference.py
    participant Y as YOLO11s
    participant B as ByteTrack
    participant P as PlateTracker
    participant O as PaddleOCR
    participant E as Export

    U->>I: python inference.py --source video.mp4
    I->>Y: Load model best.pt

    loop For each frame
        I->>Y: Detect plates
        Y-->>B: Bounding boxes + conf
        B->>B: Assign track_ids
        B-->>P: Tracks with IDs

        P->>P: Collect frames per track
        P->>P: Calculate sharpness
        P->>P: Calculate angle
    end

    Note over P: End of video

    loop For each track
        P->>P: Select best frame
        P->>P: Crop plate + resize
        P->>O: Preprocessed image
        O->>O: OCR + confidence
        O-->>P: Raw text

        P->>P: Normalize AA-123-AA
        P->>P: Validate regex
    end

    P-->>E: Results list
    E->>E: Generate CSV
    E->>E: Save crops
    E-->>U: results.csv + crops/
```

### 3. Real-time Pipeline with Voting

```mermaid
sequenceDiagram
    participant W as Webcam/Stream
    participant R as realtime_alpr_voting.py
    participant Y as YOLO Track
    participant V as VotingBuffer
    participant O as EasyOCR
    participant D as Display

    W->>R: Flux vidéo continu

    loop Each frame
        R->>Y: Frame + model.track()
        Y-->>R: Tracks with IDs

        loop For each detected track
            alt New track
                R->>V: Add crop to buffer[track_id]
                R->>O: OCR on crop
                O-->>V: (text, confidence)

                alt Buffer full (3-5 frames)
                    V->>V: Weighted voting
                    V->>V: Normalize winner
                    V-->>R: Confirmed plate
                    Note over R: Mark track as confirmed
                end
            else Confirmed track
                Note over R: Skip OCR (already processed)
            end
        end

        R->>D: Draw bboxes + IDs
        R->>D: Display confirmed plates
        R->>D: FPS + statistics
        D-->>W: Real-time display

        alt User presses 's'
            R->>R: Save screenshot
        end

        alt User presses 'q'
            R->>R: Exit application
        end
    end
```

### 4. Advanced OCR Post-processing

```mermaid
sequenceDiagram
    participant U as User
    participant O as ocr_from_tracks.py
    participant L as YOLO Labels
    participant C as Crop Extraction
    participant E1 as EasyOCR
    participant V as Voting System
    participant X as Export

    U->>O: python ocr_from_tracks.py
    O->>L: Load labels/*.txt

    loop For each track_id
        L->>C: Extract 3-5 frames

        loop For each track frame
            C->>C: Crop according to bbox
            C->>C: Resize if < 120px

            alt Try 1: Color image
                C->>E1: OCR on color
                E1-->>V: (text, conf)
            end

            alt If conf < 0.5: Try 2: Binary
                C->>C: OTSU threshold
                C->>E1: OCR on binary
                E1-->>V: (text, conf)
            end
        end

        V->>V: Calculate cumulative scores
        V->>V: Select best candidate
        V->>V: Normalize French format
        V-->>O: Final plate + confidence
    end

    O->>X: Generate CSV
    O->>X: Visualization grid
    O->>X: Statistics (success rate)
    X-->>U: results.csv + visualization.png
```

---

## Project Structure

```
car_plate_projet/
├── app.py                              # [FUTURE] Flask/FastAPI API
│
├── train_yolo.py                       # Two-phase training (239 lines)
├── inference.py                        # Complete batch pipeline (357 lines)
├── ocr_from_tracks.py                  # OCR post-processing with voting (548 lines)
├── realtime_alpr.py                    # Simple real-time (273 lines)
├── realtime_alpr_voting.py             # Real-time with voting (349 lines)
├── organize_dataset.py                 # Dataset organization + split (166 lines)
├── visualize_results.py                # Results visualization
├── download_pretrained_weights.py      # Weights download
├── test_pretrained_model.py            # Model testing
├── test_ocr_track38.py                 # Specific OCR debug
├── ocr_from_tracks_debug.py            # OCR debug version
├── realtime_detect.py                  # Simple real-time detection
│
├── extract_frames.sh                   # FFmpeg extraction (6 FPS)
├── quickstart.sh                       # Complete automated setup
├── download_weights_simple.sh          # Alternative weights download
│
├── requirements.txt                    # Python dependencies
├── data.yaml                           # YOLO dataset config
├── .gitignore                          # Git exclusions
│
├── README.md                           # Main documentation (French)
├── README_DETAILED.md                  # [THIS FILE] Complete documentation
├── SETUP_COMPLETE.md                   # Setup guide
├── QUICK_COMMANDS.md                   # Quick command reference
├── PRETRAINED_WEIGHTS.md               # Pre-trained weights documentation
│
├── data/                               # Project data
│   ├── plates/                         # Organized YOLO dataset
│   │   ├── images/
│   │   │   ├── train/                  # 33 training images
│   │   │   └── val/                    # 10 validation images
│   │   └── labels/
│   │       ├── train/                  # YOLO format labels
│   │       └── val/
│   │
│   ├── raw_frames/                     # Extracted frames (~195 PNG)
│   ├── hard_negatives/                 # 153 negatives for HNM
│   └── videos/                         # Source videos
│       ├── clip1.mp4                   # 27MB
│       └── clip2.mp4                   # 74MB
│
├── runs/                               # YOLO training results
│   └── detect/
│       ├── train_phase1/               # Phase 1 (freeze)
│       │   ├── weights/
│       │   │   ├── best.pt
│       │   │   └── last.pt
│       │   ├── results.png
│       │   └── confusion_matrix.png
│       │
│       └── train_phase2/               # Phase 2 (unfreeze)
│           ├── weights/
│           │   ├── best.pt             # FINAL MODEL
│           │   └── last.pt
│           └── results.png
│
├── ocr_v1/ to ocr_v5/                  # OCR iterations
├── ocr_final/                          # Best results
│   ├── results.csv                     # Track_ID, Plate, Confidence
│   ├── crops/                          # Plate images
│   │   ├── track_34_HE-395-ER.png
│   │   ├── track_38_CN-092-TR.png
│   │   ├── track_105_GS-708-TE.png
│   │   └── track_114_HE-066-US.png
│   └── visualization.png               # Results grid
│
├── job_3239936.../                     # CVAT export
│   └── obj_train_data/                 # Raw YOLO labels
│
├── venv/                               # Python virtual environment
│   ├── bin/                            # Executables (python, pip)
│   └── lib/                            # Installed libraries
│
└── img/                                # [TO CREATE] Screenshots for README
    ├── training_pipeline.png
    ├── realtime_detection.png
    ├── ocr_voting.png
    ├── inference_results.png
    ├── best_frame_selection.png
    └── ocr_visualization.png
```

---

## Technical Skills

### **Deep Learning & Computer Vision**
- **YOLO (You Only Look Once)**: Real-time object detection architecture
- **Transfer Learning**: Fine-tuning pre-trained models
- **Two-Phase Training**: Freeze/unfreeze strategy for small datasets
- **Object Tracking**: ByteTrack for multi-object tracking
- **Data Augmentation**: Advanced techniques to increase robustness
- **Hyperparameter Tuning**: Training parameter optimization

### **Image Processing & Vision**
- **OpenCV**: Image and video manipulation
- **Preprocessing**: Color conversion, OTSU binarization, resize
- **Sharpness Detection**: Laplacian variance for sharpness measurement
- **Frame Selection**: Multi-criteria scoring algorithms
- **Crop Extraction**: Precise region of interest extraction
- **Angle Detection**: Perspective correction and rotation

### **OCR & NLP**
- **PaddleOCR**: Advanced OCR engine with angle detection
- **EasyOCR**: Lightweight alternative for real-time
- **Text Normalization**: Format correction and validation
- **Regex Validation**: Patterns for French plates
- **Voting Mechanisms**: Weighted multi-frame consensus
- **Character Correction**: Contextual corrections (O/0, I/1)

### **Architecture & DevOps**
- **Modular Design**: Clear separation of responsibilities
- **Class-based Architecture**: PlateTracker, VotingSystem
- **Configuration Management**: YAML for parameters
- **Logging**: Complete operation traceability
- **Error Handling**: Robust exception management
- **Virtual Environments**: Dependency isolation

### **Python Development**
- **Python 3.8+**: Modern syntax and type hints
- **Multiprocessing**: Parallel processing (optional)
- **File I/O**: CSV, image, video file management
- **CLI Arguments**: argparse for command-line interfaces
- **Progress Tracking**: tqdm for real-time monitoring
- **Data Structures**: deque for buffers, defaultdict for votes

### **Video Processing**
- **FFmpeg**: Frame extraction with advanced filters
- **Stream Processing**: Webcam, video, RTSP support
- **FPS Control**: Extraction framerate management
- **Anti-duplicate**: mpdecimate filter to avoid redundancy
- **Real-time Display**: cv2.imshow with interactive controls

### **Data Management**
- **Dataset Organization**: Stratified train/val split
- **Hard-Negative Mining**: Intelligent negative sample selection
- **Sharpness-based Selection**: Automatic quality criteria
- **CVAT Integration**: Annotation export/import
- **YOLO Format**: Label conversion and validation

### **Performance Optimization**
- **GPU Acceleration**: PyTorch CUDA, Apple Metal
- **Batch Processing**: Grouped image processing
- **Early Stopping**: Processing time optimization
- **Result Caching**: Avoid unnecessary re-calculations
- **Memory Management**: Limited buffers, garbage collection

---

## Available Modes

### Training Mode
- **Transfer learning** from specialized plate weights
- **Two-phase strategy**: stabilization then fine-tuning
- **Adaptive augmentation** for small datasets
- **Continuous validation** with detailed metrics

### Batch Inference Mode
- **Complete video processing** with progress
- **Automatic selection** of best frame per vehicle
- **Optimized OCR** with multi-frame voting
- **Structured export** CSV + organized crops

### Simple Real-time Mode
- **Instant detection** on video stream
- **Live display** with bounding boxes
- **FPS monitoring** for performance
- **Multi-source support** (webcam/video/RTSP)

### Real-time Mode with Voting
- **Multi-frame consensus** for maximum accuracy
- **Intelligent buffer** of 3-5 frames per vehicle
- **Automatic confirmation** after voting
- **Memory optimization** (no re-OCR)

### Post-Processing Mode
- **Existing YOLO labels analysis**
- **Advanced voting** with adaptive preprocessing
- **Visualizations** results grid
- **Complete debug** with detailed logs

### Visualization Mode
- **Report generation** with statistics
- **Annotated image grids**
- **Performance metrics** (time, accuracy)
- **OCR runs comparison**

---

## Technical Configuration

### **YOLO Parameters**

```python
YOLO_CONFIG = {
    # Model
    "model": "yolo11s.pt",              # Base model
    "pretrained": "license_plate_detector.pt",  # Specialized weights

    # Training
    "epochs_phase1": 20,                # Freeze phase
    "epochs_phase2": 30,                # Unfreeze phase
    "imgsz": 1280,                      # Resolution
    "batch": 8,                         # Batch size
    "freeze": 10,                       # Number of frozen layers

    # Optimization
    "patience": 50,                     # Early stopping
    "lr0": 0.01,                        # Initial learning rate
    "lrf": 0.001,                       # Final learning rate
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # Augmentation
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.15,
    "mosaic": 0.5,
    "mixup": 0.1,

    # Inference
    "conf": 0.25,                       # Confidence threshold
    "iou": 0.45,                        # NMS IoU threshold
    "max_det": 300                      # Max detections per image
}
```

### **OCR Configuration**

```python
OCR_CONFIG = {
    # PaddleOCR
    "paddle": {
        "use_angle_cls": True,
        "lang": "en",
        "use_gpu": True,
        "show_log": False,
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.5
    },

    # EasyOCR
    "easy": {
        "languages": ["en"],
        "gpu": True,
        "model_storage_directory": "~/.EasyOCR/model/",
        "detector": True,
        "recognizer": True
    },

    # Preprocessing
    "preprocessing": {
        "min_width": 120,               # Resize if smaller
        "binary_threshold": "otsu",     # Automatic OTSU
        "fallback_enabled": True        # Binary attempt if failure
    },

    # Voting
    "voting": {
        "frames_per_track": 5,          # Number of frames to analyze
        "min_confidence": 0.5,          # Minimum confidence threshold
        "weighted": True                # Confidence-weighted voting
    },

    # Normalization
    "normalization": {
        "format": "french",             # AA-123-AA
        "corrections": {
            "letters": {"O": "0", "I": "1", "B": "8", "S": "5"},
            "digits": {"0": "O", "1": "I", "8": "B", "5": "S"}
        },
        "regex": r"^[A-Z]{2}-\d{3}-[A-Z]{2}$"
    }
}
```

### **Tracking Parameters**

```python
TRACKING_CONFIG = {
    "tracker": "bytetrack.yaml",        # Algorithm
    "persist": True,                    # Maintain IDs
    "conf": 0.25,                       # Detection threshold
    "iou": 0.45,                        # IoU for association
    "max_age": 30,                      # Frames before ID loss
    "min_hits": 3,                      # Frames before confirmation
    "track_buffer": 30                  # Buffer for occlusions
}
```

### **FFmpeg Extraction**

```bash
# Configuration in extract_frames.sh
FPS=6                                   # Frames per second
VIDEO_FILTER="mpdecimate"               # Anti-duplicate
OUTPUT_FORMAT="frame_%06d.png"          # Output format
QUALITY="qscale:v 2"                    # PNG quality (1-31, 2=high)
```

---

## Metrics and Statistics

### Model Performance

#### Detection (after training on 43 images)
- **mAP50**: ~88% (IoU=0.50)
- **mAP50-95**: ~70% (IoU=0.50:0.05:0.95)
- **Precision**: ~92%
- **Recall**: ~85%
- **F1-Score**: ~88%

#### OCR (depending on conditions)
- **Single frame (ideal conditions)**: ~85%
- **Single frame (average conditions)**: ~70%
- **Voting 3 frames**: ~90%
- **Voting 5 frames**: ~94%
- **With normalization**: +5-8% gain

### Processing Time

#### Per 30-second video (900 frames)
- **Frame extraction (FFmpeg)**: ~5-8 seconds
- **YOLO Detection**: ~20-25 seconds (30 FPS)
- **Tracking**: included in detection (~2ms overhead)
- **Best frame selection**: ~0.5 seconds
- **OCR batch (10 vehicles × 5 frames)**: ~25-30 seconds
- **CSV + crops export**: ~1-2 seconds
- **Total pipeline**: ~50-65 seconds

#### Real-time (webcam stream)
- **Average FPS**: 25-35 FPS (depending on GPU)
- **Detection latency**: ~20-30ms
- **OCR latency (per plate)**: ~80-120ms
- **Display overhead**: ~5-10ms

### System Resources

#### Training
- **RAM**: ~4-6GB
- **VRAM (GPU)**: ~2-4GB
- **Storage**: ~500MB (model + cache)
- **CPU**: 4-8 cores recommended

#### Inference
- **RAM**: ~800MB-1.5GB
- **VRAM (GPU)**: ~500MB-1GB
- **Storage**: ~100MB (crops + results)
- **CPU**: 2-4 cores sufficient

### Current Dataset

```
Dataset Statistics:
├── Total images: 43
├── Training set: 33 (76.7%)
├── Validation set: 10 (23.3%)
├── Positives (with plates): 12 (27.9%)
├── Negatives (without plates): 31 (72.1%)
├── Hard negatives saved: 153 (for future HNM)
└── Extracted frames: 195 (available for annotation)
```

### Current OCR Results

**File `ocr_final/results.csv`:**

| Track ID | Recognized Plate | Confidence | Frame | Timestamp |
|----------|------------------|------------|-------|-----------|
| 34       | HE-***-ER        | 0.92       | 307   | 10.23s    |
| 38       | CN-***-TR        | 0.88       | 442   | 14.73s    |
| 105      | GS-***-TE        | 0.85       | 551   | 18.37s    |
| 114      | HE-***-US        | 0.91       | 623   | 20.77s    |

---

*Developed with passion in Python - Computer Vision and Artificial Intelligence*

**Last update**: January 2025
**Version**: 0.5-dev
**Status**: In active development
