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
practice-group-5
│
├── src
│   ├── checker.py
│   └── visualizer.py
│
├── app.py
├── requirements.txt
├── docs
└── README.md
```


## Installation

```bash
git clone https://github.com/BISTU-OSSD/practice-group-5.git

cd practice-group-5

pip install -r requirements.txt
```


## Usage

### Dataset Checking

Run the YOLO dataset checker:

```bash
python src/checker.py
```

Input your YOLO dataset path:

```text
demo_dataset
```

The checker will analyze:

- Missing image-label pairs
- Orphan label files
- Empty label files
- Invalid bounding box annotations
- Annotation format errors


### Visualization

Run the visualization module:

```bash
python src/visualizer.py
```

The visualization module can display YOLO annotations on images.


## Development Workflow

This project uses GitHub Flow for collaborative development.

```text
main
 |
dev
 |
feature branches
```

Development process:

1. Create a new feature branch from `dev`.

Example:

```bash
git checkout dev

git checkout -b feature/your-feature-name
```

2. Implement your changes.

3. Commit your changes:

```bash
git add .

git commit -m "Add new feature"
```

4. Push your branch:

```bash
git push origin feature/your-feature-name
```

5. Create a Pull Request to merge into `dev`.

6. After review, merge changes into the development branch.


## Branches

Current development branches:

- `main` - Stable release branch
- `dev` - Development integration branch
- `feature/checker` - YOLO dataset checking module
- `feature/visualizer` - Annotation visualization module
- `feature/statistics` - Dataset statistics module
- `feature/web` - Web interface module
- `feature/tests-docs` - Testing and documentation module


## License

This project is released under the MIT License.
