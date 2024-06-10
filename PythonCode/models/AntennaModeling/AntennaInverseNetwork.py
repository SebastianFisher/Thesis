import torch.nn as nn
import torch

# Neural Network designed to create image for inputted number
class AntennaInverseNN(nn.Module):
    
    def __init__(self):
        super(AntennaInverseNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(81, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 12*12),
        )
        self.Tanh = nn.Tanh()
        # self.m = m
    
    def forward(self, x, m, use_threshold=False):
        
        output = self.network(x)
        
        # use "soft thresholding" (ST) function from this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10130105&tag=1
        # f(x) = 1/2 + 1/2 * tanh(m(x - 1/2)), where m is the thresholding hyperparameter
        # when m = 1, it is "a generalization of the unipolar sigmoid function"?
        # this should make the network ideally output mostly 0's and 1's, which is what we want the structure to be...

        if use_threshold:
            thresholded_output = 1/2 * (self.Tanh(m * (output - 1/2)) + 1)
            return thresholded_output

        return output
        # return torch.sigmoid(output)