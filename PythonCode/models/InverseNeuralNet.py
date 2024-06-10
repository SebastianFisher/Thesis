import torch.nn as nn
# Neural Network designed to create image for inputted number
class ReverseNN(nn.Module):

  def __init__(self):
    super().__init__()

    self.linear_relu_stack = nn.Sequential(
        nn.Linear(10, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 28*28),
    )

  def forward(self, x):
    # x = self.flatten(x)
    output = self.linear_relu_stack(x)
    return output

