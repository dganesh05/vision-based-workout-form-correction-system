# Vision-Based Workout Form Correction System

Group 3 | ITCS 4152

Vision-based workout form correction project focused on safer strength training through pose estimation and biomechanics analysis. The long-term objective is to move from 2D keypoints to 3D form understanding and generate actionable feedback such as "go deeper" or "keep knees aligned".

## Table of Contents

- [Who This README Is For](#who-this-readme-is-for)
- [Project Status](#project-status)
- [Problem and Motivation](#problem-and-motivation)
- [System Pipeline](#system-pipeline)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Current Usage (Implemented)](#current-usage-implemented)
- [Roadmap Usage (Planned)](#roadmap-usage-planned)
- [Data Collection and Dataset Design](#data-collection-and-dataset-design)
- [Biomechanical Features and Analysis](#biomechanical-features-and-analysis)
- [Reproducibility Notes](#reproducibility-notes)
- [Evaluation Plan](#evaluation-plan)
- [Contributing](#contributing)
- [Citation and Acknowledgments](#citation-and-acknowledgments)
- [License](#license)

## Who This README Is For

This README is intentionally structured for multiple audiences:

- Users and integrators who want to run the project and adapt it into larger systems.
- Researchers and students who want to reproduce the pipeline and compare methods.
- Contributors who want to add new models, features, and datasets.
- Evaluators and instructors who need a clear separation between implemented functionality and planned work.

## Project Status

The project currently contains an early implementation of 2D pose extraction and a defined technical roadmap for full 3D workout form correction.

| Pipeline Stage | Status | Notes |
| --- | --- | --- |
| Video input processing | Implemented (basic) | Single-view workflow in notebook form. |
| 2D pose extraction (YOLOv8-Pose) | Implemented (initial) | Available via notebook workflow. |
| 3D pose lifting (TGMF-Pose / GLA-GCN / VideoPose3D) | Planned / In progress | Methodology defined; integration pending. |
| Feature extraction (angles, symmetry, depth) | Planned | Target features identified. |
| Anomaly detection (Mahalanobis + thresholding/ML) | Planned | Design defined; implementation pending. |
| User feedback generation | Planned | Feedback taxonomy defined. |

## Problem and Motivation

Incorrect lifting form is a common source of preventable training injuries. Most lightweight consumer systems rely only on 2D tracking, which can miss depth-dependent biomechanical issues such as:

- Knee valgus
- Insufficient squat depth
- Trunk lean and postural collapse

This project addresses those limitations by combining robust pose estimation with planned 3D reconstruction and temporal biomechanical analysis.

## System Pipeline

The target architecture is:

Video Input -> 2D Pose -> 3D Pose -> Feature Extraction -> Analysis -> Feedback

### Stage 1: Data Collection

- Bodyweight squat videos are captured using smartphone + tripod setups.
- Multi-view design uses up to 8 recording angles for stronger geometric coverage.
- Subject diversity target includes height, weight, gender, and lifting experience variation.
- Correct-form references are aligned against gym-guided baseline biomechanics.

### Stage 2: Pose Estimation and 3D Lifting

- 2D extraction model: YOLOv8-Pose.
- Annotation tooling: Labelme (for custom/edge-case labeling).
- Planned 3D lifting models:
	- TGMF-Pose
	- GLA-GCN
	- VideoPose3D
- Heavy training/inference workloads are designed for HPC execution using Python + PyTorch.

### Stage 3: Feature Extraction and Temporal Analysis

- Temporal smoothing across multiple frames for robust motion understanding.
- Planned sequence alignment with Dynamic Time Warping (DTW) to normalize speed differences.
- Joint kinematics and symmetry features computed from reconstructed pose trajectories.

### Stage 4: Anomaly Detection and Feedback

- Planned Mahalanobis-distance-based outlier scoring across angle-feature vectors.
- Hybrid rule-based + learned thresholds for form classification.
- User-facing feedback examples:
	- Go deeper
	- Keep knees aligned
	- Maintain upright posture

## Repository Structure

- README.md: Project overview, usage, reproducibility, and contribution guidance.
- readme_details: Detailed architecture and methodology notes used to build this README.
- extract_2d_pose.ipynb: Notebook for current 2D extraction workflow.
- requirements.txt: Current Python dependencies.
- LICENSE: MIT license.

## Quick Start

### 1. Clone and enter project

```bash
git clone <your-fork-or-repo-url>
cd vision-based-workout-form-correction-system
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch notebook workflow

```bash
jupyter notebook
```

Open extract_2d_pose.ipynb and run cells in order.

## Current Usage (Implemented)

Current repository functionality is centered on 2D pose extraction.

1. Prepare an input video path in extract_2d_pose.ipynb.
2. Run notebook cells sequentially.
3. Inspect generated keypoint outputs and visual overlays.

Notes:

- The notebook is the primary entry point in the current codebase.
- The full 3D correction pipeline is not fully wired in this repository yet.

## Roadmap Usage (Planned)

Planned usage flow after full integration:

1. Ingest user squat video.
2. Run 2D keypoint extraction.
3. Lift to 3D pose sequence.
4. Compute temporal biomechanical features.
5. Classify deviations from correct form.
6. Return interpretable correction feedback.

## Data Collection and Dataset Design

Planned/ongoing dataset design principles:

- Exercise focus: bodyweight squats (initial scope).
- Multi-angle captures: up to 8 camera viewpoints.
- Population size target: 15-20 participants.
- Demographic and anthropometric diversity encouraged.
- Ground truth assisted by gym-based reference practice.

If you are contributing data, include metadata for viewpoint, subject profile category, and labeling quality notes.

## Biomechanical Features and Analysis

Core features targeted for correction quality:

- Knee angle
- Hip angle
- Torso alignment angle
- Squat depth
- Left-right symmetry

These features are intended to be aggregated over time, not only at single frames, to improve stability.

## Reproducibility Notes

To make experiments easier to reproduce:

- Keep raw videos and processed outputs in stable, documented folder layouts.
- Record model versions and checkpoint sources when adding 3D lifting experiments.
- Track preprocessing settings (frame rate, resizing, keypoint confidence thresholds).
- Log experiment parameters and evaluation results in a run sheet.

Recommended future folders (to be created as implementation expands):

- data/raw
- data/processed
- outputs/poses_2d
- outputs/poses_3d
- outputs/analysis
- experiments

## Evaluation Plan

Evaluation will compare predicted form quality against validated references and include:

- Per-feature error or deviation scoring.
- Form anomaly detection quality.
- Consistency across camera angles and subject variation.
- Actionability of feedback.

Current repository state:

- Evaluation scripts and benchmark tables are not yet published here.
- This section should be updated as soon as metrics are available.

## Contributing

Contributions are welcome from engineering, ML, and biomechanics perspectives.

### Suggested contribution workflow

1. Open an issue describing the proposed change.
2. Fork and create a focused branch.
3. Add or update code/notebooks and documentation.
4. Include reproducibility notes for new experiments.
5. Open a pull request with a clear summary and test evidence.

### High-impact contribution areas

- 3D lifting model integration (TGMF-Pose, GLA-GCN, VideoPose3D)
- Feature extraction implementations
- Form-classification and feedback logic
- Dataset tooling and annotation pipelines
- Evaluation and visualization tooling

A dedicated CONTRIBUTING.md can be added in a future update; until then, use this section as the baseline process.

## Citation and Acknowledgments

If this project supports your work, please cite it in your reports/papers once a formal citation format is published.

Acknowledgments:

- ITCS 4152 course support
- Group 3 team members
- Open-source pose estimation research communities

## License

This project is licensed under the MIT License. See the LICENSE file for details.