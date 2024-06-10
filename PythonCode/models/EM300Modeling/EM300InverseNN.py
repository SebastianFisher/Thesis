import torch.nn as nn
import torch

# Neural Network designed to create image for inputted number
class EM300InverseNN(nn.Module):
    def __init__(self):
        super(EM300InverseNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(54, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 16*16),
        )
        self.Tanh = nn.Tanh()
    
    def forward(self, x, m, use_threshold=True):
        # get 16x16 inner part of struct from network
        output_inner = self.network(x)
        if output_inner.dim() == 1:  # if no batch
            output_inner = output_inner.reshape([1,1,16,16])
        output_inner = output_inner.reshape([output_inner.shape[0], 1, 16, 16])
        
        # use "soft thresholding" (ST) function from this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10130105&tag=1
        # f(x) = 1/2 + 1/2 * tanh(m(x - 1/2)), where m is the thresholding hyperparameter
        # this should help the network ideally output mostly 0's and 1's, which is what we want the structure to be...

        if use_threshold:
            output_inner = 1/2 * (self.Tanh(m * (output_inner - 1/2)) + 1)
        
        output = torch.zeros([output_inner.shape[0], output_inner.shape[1], 18, 18]).to(output_inner.device)
        # hard code ports
        # (8,0),(8,17),(17,7),(0,7) Left Right Bottom Top
        # output_struct = torch.zeros([output.shape[0], output.shape[1], 18, 18], requires_grad=False).to(output.device)
        output[:,:,8,0] = 1
        output[:,:,8,17] = 1
        output[:,:,17,7] = 1
        output[:,:,0,7] = 1
        output[:,:,1:17,1:17] += output_inner
        
        return output
