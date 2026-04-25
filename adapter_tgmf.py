import torch
import numpy as np

# TODO: Import the specific model class and tokenizer from the cloned TGMF-Pose repository.
# e.g., from models.tgmf import TGMFOcclusionNet
# e.g., from transformers import CLIPTokenizer, CLIPTextModel

def run_tgmf_pose_inference(sequence_2d: np.ndarray, checkpoint: str, device: str, prompt: str) -> np.ndarray:
    """
    Adapter for TGMF-Pose inference using text-guided temporal lifting.
    sequence_2d: NumPy array of shape (T, 17, 3) containing (x, y, confidence)
    """
    torch_device = torch.device(device)
    T, V, C = sequence_2d.shape

    # 1. Coordinate Normalization (Center on Pelvis)
    normalized_sequence = np.copy(sequence_2d)
    for t in range(T):
        pelvis = (sequence_2d[t, 11, :2] + sequence_2d[t, 12, :2]) / 2.0
        normalized_sequence[t, :, :2] = sequence_2d[t, :, :2] - pelvis

    # 2. Convert to PyTorch Tensor
    tensor_2d = torch.tensor(normalized_sequence, dtype=torch.float32).to(torch_device)
    tensor_2d = tensor_2d.permute(2, 0, 1).unsqueeze(0) # Shape: (1, 3, T, 17)

    # 3. Text Prompt Encoding
    # TODO: Pass the prompt through the text encoder used by the repo (usually CLIP)
    text_embedding = None 
    # e.g., tokens = tokenizer(prompt, return_tensors="pt")
    #       text_embedding = text_model(**tokens).last_hidden_state.to(torch_device)

    # 4. Model Initialization & Weight Loading
    # TODO: Initialize the specific model class
    model = None # e.g., model = TGMFOcclusionNet(num_joints=17)
    
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=torch_device))
    model.to(torch_device)
    model.eval()

    # 5. Inference
    with torch.no_grad():
        # TODO: Pass both the kinematics and the text embeddings into the model
        output_3d = model(tensor_2d, text_features=text_embedding)
        
    # 6. Reformat back to standardized NumPy array (T, 17, 3)
    output_3d = output_3d.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    return output_3d