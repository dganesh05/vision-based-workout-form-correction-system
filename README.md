# Vision-Based Workout Form Correction System

## Branch: feedback-transformer

This branch contains the full squat analysis and feedback pipeline for our Computer Vision project.

The goal is to automatically evaluate squat form using pose estimation, angle extraction, machine learning classification, and personalized feedback generation.

---

## What I Worked On

I focused mainly on:

- rep-by-rep squat detection
- angle extraction and biomechanics analysis
- training dataset creation
- machine learning model training
- prediction system
- personalized feedback engine

This branch handles the full feedback + ML pipeline.

---

## Project Flow

### Step 1 — Video Input

User records a squat video from front, back, or side angle.

↓

### Step 2 — MotionBERT / Pose Estimation

MotionBERT extracts 3D body keypoints from the video.

This gives:

- body joint coordinates
- frame-by-frame pose landmarks

↓

### Step 3 — Angle Extraction (`compute_angles.py`)

Using the keypoints, we calculate:

- Right Knee Angle
- Left Knee Angle
- Right Hip Angle
- Left Hip Angle
- Average Knee Angle
- Average Hip Angle
- Spine Lean
- Knee Symmetry

The script also detects squat reps using the bottom position of each squat (local minimum knee angle).

Output:

final_features.csv

↓

### Step 4 — Build Training Dataset (`build_training_data_v2.py`)

This combines:

- data/angles_csv/correct
- data/angles_csv/incorrect

and creates the final rep-by-rep training dataset.

Output:

training_data_v2.csv

Current dataset size:

- 131 total reps

↓

### Step 5 — Model Training (`train_model_v2.py`)

We train a Random Forest Classifier using:

Features:

- avg_knee_angle
- avg_hip_angle
- spine_lean
- knee_symmetry

Important:

No hardcoded rule-based labels are used for model training.

The model learns from reviewed squat data instead of fixed threshold logic.

Final Model Performance:

- Accuracy: 88.89%

Feature Importance:

1. Knee Symmetry
2. Hip Angle
3. Knee Angle
4. Spine Lean

This means balance and symmetry were the strongest predictors.

Output:

squat_model.pkl

↓

### Step 6 — Prediction System (`predict.py`)

This loads the trained model and predicts squat quality for new squat inputs.

Prediction classes:

- Excellent Squat
- Good Squat + Minor Improvements
- Needs Major Improvement

It also gives explainable feedback like:

- Go deeper
- Improve balance
- Keep chest up
- Improve hip positioning

This acts as the final AI Personal Squat Coach.

---

## Files Added / Updated

### Core Files

- compute_angles.py
- build_training_data.py
- build_training_data_v2.py
- feedback_engine.py
- train_model.py
- train_model_v2.py
- predict.py

### CSV Outputs

- final_features.csv
- final_feedback_results.csv
- training_data.csv
- training_data_v2.csv

### Model File

- squat_model.pkl

---

## How To Review My Work

### 1. Pull this branch

git checkout feedback-transformer
git pull origin feedback-transformer

---

### 2. Build training dataset

python feedback/build_training_data_v2.py

This generates:

training_data_v2.csv

---

### 3. Train the model

python feedback/train_model_v2.py

This trains the Random Forest model and saves:

squat_model.pkl

---

### 4. Run prediction demo

python feedback/predict.py

This shows final squat prediction + personalized feedback output.

---

## Final Goal

Final system should work like:

Video Upload
→ MotionBERT
→ 3D Keypoints
→ Angle Extraction
→ Model Prediction
→ Feedback Engine
→ Final Squat Coaching Output

---

## Notes

- feedback logic is explainable and biomechanics-based
- model training avoids hardcoded threshold labels
- real squat videos are used for validation
- more video testing will improve final accuracy

