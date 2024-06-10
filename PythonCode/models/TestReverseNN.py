from InverseNeuralNet import ReverseNN
import matplotlib.pyplot as plt
import torch

MODELS_PATH = "./saved/"

# Load reverse network first, 
reverse_model = ReverseNN()
reverse_model.load_state_dict(torch.load(MODELS_PATH + "MNISTInverseNet"))
reverse_model.eval()

# Try to generate a handwritten digit
digit = 7
test_7 = torch.reshape(reverse_model(torch.tensor([1 if i == digit - 1 else 0 for i in range(10)], dtype=torch.float32)), [28,28])

digit = 9
test_9 = torch.reshape(reverse_model(torch.tensor([1 if i == digit - 1 else 0 for i in range(10)], dtype=torch.float32)), [28,28])

img = test_7.detach().numpy()
img2 = test_9.detach().numpy()

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(img2, cmap="gray")
