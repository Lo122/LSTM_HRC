import torch
import torch.nn as nn
import numpy as np

# ============================================================
#load pth 
def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model