import torch
import numpy as np

# TODO: Import the specific model class from the cloned GLA-GCN repository.
# e.g., from lib.model.gla_gcn import GLAGCN

def run_glagcn_inference(sequence_2d: np.ndarray, checkpoint: str, device: str) -> np.ndarray:
    """
    Adapter for GLA-GCN inference.
    sequence_2d: NumPy array of shape (T, 17, 3) containing (x, y, confidence)
    """
    torch_device = torch.device(device)
    T, V, C = sequence_2d.shape

    # 1. Coordinate Normalization (Center on Pelvis)
    # COCO Hips: 11 (Left Hip), 12 (Right Hip)
    normalized_sequence = np.copy(sequence_2d)
    
    for t in range(T):
        left_hip = sequence_2d[t, 11, :2]
        right_hip = sequence_2d[t, 12, :2]
        pelvis = (left_hip + right_hip) / 2.0
        
        # Subtract pelvis (x,y) from all joints, but leave the confidence score alone
        normalized_sequence[t, :, :2] = sequence_2d[t, :, :2] - pelvis

    # 2. Convert to PyTorch Tensor
    # Research models often expect shape (Batch, Channels, Time, Vertices) -> (1, 3, T, 17)
    tensor_2d = torch.tensor(normalized_sequence, dtype=torch.float32).to(torch_device)
    tensor_2d = tensor_2d.permute(2, 0, 1).unsqueeze(0) 

    # 3. Model Initialization & Weight Loading
    # TODO: Initialize the specific model class based on the repo's parameters
    model = None # e.g., model = GLAGCN(num_joints=17, in_channels=3)
    
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=torch_device))
    model.to(torch_device)
    model.eval()

    # 4. Inference
    with torch.no_grad():
        # TODO: Call the model's forward pass. It should output 3D coordinates.
        output_3d = model(tensor_2d) # Output expected shape: (1, 3, T, 17)
        
    # 5. Reformat back to standardized NumPy array (T, 17, 3)
    output_3d = output_3d.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    return output_3d