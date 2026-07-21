# YOLO Dataset Inspector

YOLO Dataset Inspector is an open-source tool for checking and visualizing YOLO object detection datasets.

## Introduction

YOLO models require high-quality annotated datasets.
However, manually checking annotation errors is time-consuming.

This project provides automatic dataset inspection and visualization functions to help developers quickly analyze YOLO-format datasets.

## Features

- Check image-label matching
- Detect invalid YOLO annotations
- Verify bounding box format
- Visualize annotation results
- Analyze dataset statistics

## Project Structure

```text
YOLO-Dataset-Inspector
│
├── src
│   ├── checker.py
│   └── visualizer.py
│
├── app.py
├── requirements.txt
└── docs
```

## Installation

```bash
git clone https://github.com/oooljn/YOLO-Dataset-Inspector.git

cd YOLO-Dataset-Inspector

pip install -r requirements.txt
Environment

Python >=3.10

Pillow

pytest

Testing
bash
python -m pytest tests/ -v
Usage Example
python
from src.visualizer import visualize
visualize("examples/demo_dataset/images/good.jpg",
          "examples/demo_dataset/labels/good.txt",
          "tests/output/vis.jpg")