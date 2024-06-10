import onnx
import torch
import numpy as np
from onnx2torch import convert

# Path to ONNX model
onnx_model_path = '../sparam_cnn.onnx'
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)

# Or you can load a regular onnx model and pass it to the converter
#onnx_model = onnx.load(onnx_model_path)
#torch_model_2 = convert(onnx_model)

# Evaluate model on mock data
mock_struct = torch.ones((18,18))

print(torch_model_1)

