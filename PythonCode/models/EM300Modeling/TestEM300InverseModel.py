from EM300InverseNN import EM300InverseNN
from EM300CNN import EM300CNN
import torch
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import json

RANDOM_SEED = 10
torch.manual_seed(RANDOM_SEED)

# Create datasets
######################################################################
# Dataset object for DataLoader
class EM300Dataset(Dataset):
    def __init__(self, x_data_path, y_data_path):
        super().__init__()
        self.x = torch.tensor(np.load(x_data_path))
        self.y = torch.tensor(np.load(y_data_path))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].unsqueeze(0), self.y[idx]

# Load datasets
DATA_DIR = "/scratch/gpfs/sbfisher/EM300/data/"
# Create datasets for training the inverse model. 
# Use the training dataset from forward network
x_data_path = DATA_DIR + "x_data.npy"
y_data_path = DATA_DIR + "y_data.npy"

dataset = EM300Dataset(x_data_path, y_data_path)

generator = torch.Generator().manual_seed(RANDOM_SEED)
train, val, test = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator=generator)
# batch_size = 512
batch_size = 4096

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, generator=generator)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, generator=generator)
######################################################################


# Set up forward and reverse models
######################################################################
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# device = "cpu"
print(f"Using {device} device")

MODELS_PATH = "./saved/"

# Load pretrained model
forward_model = EM300CNN(mean=0.4068).to(device)
forward_model.load_state_dict(torch.load("./saved/EM300CNNTunedLR0.001_gamma0.8_Every5_E50_Bs_4096", map_location=device))
forward_model.eval()

# inverse_model = AntennaInverseNN(m=20, use_threshold=True).to(device)
inverse_model = EM300InverseNN().to(device)
# MODEL_NAME = "EM300InverseNN_E40_Bstart10_Decay0.8_Step5_NewModel"
MODEL_NAME = "EM300InverseNN_E50_Bstart10_BIncr20_Decay0.85_Step5_NewModel"
inverse_model.load_state_dict(torch.load(MODELS_PATH + MODEL_NAME, map_location=torch.device(device)))
inverse_model.eval()
######################################################################


# Train Inverse Model
######################################################################
print('Training....')
total = 0
correct = 0
start = time.time()


for name, child in forward_model.named_children():
    for _, param in child.named_parameters():
        param.requires_grad = False
    if "batchnorm" in name:
        child.running_mean.requires_grad = False
        child.running_var.requires_grad = False

# optimizer = torch.optim.Adam(inverse_model.parameters(), lr=0.001)#, betas=(1,1))
# try smaller learning rate
lr = 0.001
optimizer = torch.optim.RAdam(inverse_model.parameters(), lr=lr)
step_size=5
gamma=0.8
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# loss_fn = nn.MSELoss()
loss_fn = nn.L1Loss()

weight_Ls = 1
weight_Lb = 1

print("Testing on test set")
# test on validation set
inverse_model.eval()
running_loss = [0,0]
# Use val loader for hyperparameter tuning of hard cut off threshold
data_loader = val_loader
thresholds = [round(0.1 + x * 0.05, 2) for x in range(int((0.9 - 0.1) / 0.05) + 1)]
thresholds = [0.35]
if len(thresholds) > 1:
    print("Running validation for threshold values: " + str(thresholds))
else:
    print("Running test with threshold: " + str(thresholds[0]))
    data_loader = test_loader

val_losses = []
with torch.no_grad():
    for threshold in thresholds:
        running_loss = 0
        for data in data_loader:
            struct, spect = data
            struct = struct.to(device).float()
            spect = spect.to(device).float()
            
            m=1
            output = inverse_model(spect, m=m, use_threshold=True)
        
            # For test loss, we use a hard thresholding function before computing loss
            # since designs must be either 1's (metal) or 0's (no metal)
            output = torch.where(output > threshold, 1, 0)
            
            # get predicted spectrum from forward model using normalized output
            pred_spect = forward_model(output)
        
            # Compute loss
            # S = true spectrum, S' = pred spectrum, D = design, D' = pred design
            # MSE(S, S') + MSE(D, D') + binary loss of each squre in pred design
            # for_loss = output.flatten(start_dim=1)
            # binary_loss = torch.mean(torch.square(for_loss * (for_loss - 1)))
            
            L_s = loss_fn(pred_spect, spect)
            # L_d = loss_fn(output, struct)
            # L_b = binary_loss
        
            # loss = weight_Ls * L_s # + weight_Ld * L_d + weight_Lb * L_b
        
            # scale loss by batch size
            running_loss += L_s.item() * spect.shape[0]
    
        running_loss /= len(data_loader.dataset)
        if len(thresholds) > 1:
            val_losses.append(running_loss)
            print("Validation loss for threshold={} is {}".format(threshold, running_loss))
        else:
            print("Test loss for threshold={} is {}".format(threshold, running_loss))
    
######################################################################
# Save losses to a file
if len(thresholds) > 1:
    loss_file = "/scratch/gpfs/sbfisher/EM300/inverse_loss/inverse_val_losses.npy"
    np.save(loss_file, np.array(val_losses))
    print("validation losses saved to file: " + loss_file)





