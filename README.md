## How To Test / Access My Work (BiLSTM Baseline Model)

### Step 1 — Switch to this branch

git checkout bilstm-baseline-model  
git pull origin bilstm-baseline-model  

This branch contains the clean baseline model using a BiLSTM Autoencoder + rule-based feedback system.

---

### Step 2 — Build the dataset

Run:

python feedback/build_autoencoder_dataset.py  

This script:

- loads squat `.npy` files from  
  data/model_ready_reps/  
- separates:
  - golden reference squats (correct)
  - original user squats (test data)
- reshapes data from (41, 7, 3) → (41, 21)
- applies MinMax normalization (fit only on golden data)

Outputs:

- feedback/X_golden_train.npy  
- feedback/X_user_test.npy  
- feedback/golden_scaler.pkl  

---

### Step 3 — Train the BiLSTM Autoencoder

Run:

python feedback/train_bilstm_autoencoder.py  

This trains the model using only correct squat data.

Model learns:

- ideal squat motion patterns  
- temporal movement behavior  

Saved outputs:

- feedback/final_bilstm_autoencoder_3d.keras  
- feedback/autoencoder_loss_curve_3d.png  

---

### Step 4 — Run prediction and feedback

Run:

python feedback/predict_autoencoder_quality.py  

This:

- loads trained model + scaler  
- runs prediction on user squat data  
- reconstructs input sequences  
- calculates:
  - reconstruction error  
  - max deviation  

Then applies rule-based logic to generate:

- squat quality classification  
- failure type (depth, valgus, lean, etc.)  
- performance score (0–100)  
- personalized feedback  

Saved output:

- feedback/final_prediction_results.csv  

---

### Step 5 — Review important files

Main pipeline files:

- feedback/build_autoencoder_dataset.py  
- feedback/train_bilstm_autoencoder.py  
- feedback/predict_autoencoder_quality.py  

Main outputs:

- feedback/X_golden_train.npy  
- feedback/X_user_test.npy  
- feedback/final_bilstm_autoencoder_3d.keras  
- feedback/final_prediction_results.csv  
- feedback/autoencoder_loss_curve_3d.png  

---

### Key Idea

This model does NOT directly classify squats.

Instead:

- learns correct squat patterns  
- detects deviations using anomaly detection  
- converts deviations into meaningful feedback  

---

### Example Output

Failure Type: Knee Valgus ❌  
Performance Score: 67  
Level: Beginner+  
Feedback: Improve knee tracking  

---

### Summary

This baseline model provides:

- anomaly detection for squat quality  
- interpretable biomechanical feedback  
- a strong foundation for the final model (Bi-CGRU with Attention)
