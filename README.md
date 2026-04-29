## How To Test / Access My Work

### Step 1 — Switch to this branch

git checkout feedback-transformer
git pull origin feedback-transformer

This branch contains the full feedback + ML pipeline.

---

### Step 2 — Build the training dataset

Run:

python feedback/build_training_data_v2.py

This combines:

- correct squat files
- incorrect squat files

from:

data/angles_csv/

and creates:

training_data_v2.csv

This is the final rep-by-rep training dataset.

---

### Step 3 — Train the model

Run:

python feedback/train_model_v2.py

This trains the Random Forest model using:

- knee angle
- hip angle
- spine lean
- knee symmetry

and saves:

squat_model.pkl

You will also see:

- accuracy
- confusion matrix
- feature importance

Final model accuracy is around:

88–90%

---

### Step 4 — Run final prediction demo

Run:

python feedback/predict.py

This loads the trained model and predicts squat quality for a new squat sample.

It outputs:

- squat classification
- improvement areas
- personalized coaching feedback

Example:

Excellent Squat
Go slightly deeper
Improve balance

This is the final demo output.

---

### Step 5 — Review important files

Main files to check:

- feedback/compute_angles.py
- feedback/build_training_data_v2.py
- feedback/train_model_v2.py
- feedback/predict.py
- feedback/feedback_engine.py

Main outputs:

- feedback/final_features.csv
- feedback/final_feedback_results.csv
- feedback/training_data_v2.csv
- feedback/squat_model.pkl

These contain the full final workflow.