from EM300InverseNN2 import EM300InverseNN   # Model 2 has more hidden layers



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
input("enter to continue and start training")

MODELS_PATH = "./saved/"

# Load pretrained model
forward_model = EM300CNN(mean=0.4068).to(device)
forward_model.load_state_dict(torch.load("./saved/EM300CNNTunedLR0.001_gamma0.8_Every5_E50_Bs_4096", map_location=device))
forward_model.eval()

# inverse_model = AntennaInverseNN(m=20, use_threshold=True).to(device)
inverse_model = EM300InverseNN().to(device)

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
gamma=0.85
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# loss_fn = nn.MSELoss()
loss_fn = nn.L1Loss()

hyperparam_values = {"L_s": [1], "L_d": [0], "L_b": [0]}

# try adding binary loss after 5 epochs
bin_loss_start = 10
incr_bin_loss = 20
incr_bin_loss_again = 30
weight_Lb = 0
weight_Ls = 1

print(f"Training dataset size: {len(train_loader.dataset)}")
try:
    epochs = 50

    loss_v_epoch = []
    val_loss_v_epoch = []
    # Run training for this set of hyperparams
    for epoch in range(epochs):
        
        for i, data in enumerate(train_loader, 1):
            struct, spect = data
            struct = struct.to(device).float()
            spect = spect.to(device).float()
            
            # if(torch.cuda.is_available()):
            #     struct, result = labels.cuda()
            optimizer.zero_grad()
            
            m=1
            output = inverse_model(spect, m=m, use_threshold=True)
            
            # get predicted spectrum from forward model using normalized output
            pred_spect = forward_model(output)

            # start binary loss after roughly 10 epochs
            # increase weight to 10 after 10 more epochs
            # then to 40 10 epochs after that
            if epoch >= incr_bin_loss_again:
                weight_Lb = 40
            elif epoch >= incr_bin_loss:
                weight_Lb = 10
            elif epoch >= bin_loss_start:
                weight_Lb = 1

            # Compute loss
            # S = true spectrum, S' = pred spectrum, D = design, D' = pred design
            # MSE(S, S') + MSE(D, D') + binary loss of each squre in pred design
            for_loss = output.flatten(start_dim=1)
            binary_loss = torch.mean(torch.square(for_loss * (for_loss - 1)))
            
            L_s = loss_fn(pred_spect, spect)
            # L_d = loss_fn(output, struct)
            L_b = binary_loss

            loss = weight_Ls * L_s + weight_Lb * L_b
            # loss_components = {"Ls": L_s, "Lb": L_b}
            
            if(i%20 == 0):
                print('Epoch: {} Batch: {} loss: {}'.format(epoch, i, loss.item()))
                print("Loss Breakdown: Ls: {}, Lb: {}".format(float(L_s), float(L_b)))
                print(f"Ls/Lb = {round(float(L_s)/float(L_b), 6)}")
                print("Weight_Lb = {}".format(weight_Lb))
                
            # Backpropagate gradients
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
        print("Testing on training set")
        # test on validation set
        inverse_model.eval()
        running_loss = [0,0]
        for data in train_loader:
            struct, spect = data
            struct = struct.to(device).float()
            spect = spect.to(device).float()
            
            m=1
            output = inverse_model(spect, m=m, use_threshold=True)
            
            # get predicted spectrum from forward model using normalized output
            pred_spect = forward_model(output)

            # Compute loss
            # S = true spectrum, S' = pred spectrum, D = design, D' = pred design
            # MSE(S, S') + MSE(D, D') + binary loss of each squre in pred design
            for_loss = output.flatten(start_dim=1)
            binary_loss = torch.mean(torch.square(for_loss * (for_loss - 1)))
            
            L_s = loss_fn(pred_spect, spect)
            # L_d = loss_fn(output, struct)
            L_b = binary_loss

            loss = weight_Ls * L_s + weight_Lb * L_b

            # scale loss by batch size
            running_loss[0] += L_s.item() * spect.shape[0]
            running_loss[1] += L_b.item() * spect.shape[0]

        # track loss over validation set
        running_loss[0] /= len(train_loader.dataset)
        running_loss[1] /= len(train_loader.dataset)
        loss_v_epoch.append(running_loss)
        print("Training Loss, (L_s, L_b): {}".format(loss_v_epoch[epoch]))
        
        print("Testing on validation set")
        # test on validation set
        inverse_model.eval()
        running_loss = [0,0]
        for data in val_loader:
            struct, spect = data
            struct = struct.to(device).float()
            spect = spect.to(device).float()
            
            m=1
            output = inverse_model(spect, m=m, use_threshold=True)
            
            # get predicted spectrum from forward model using normalized output
            pred_spect = forward_model(output)

            # Compute loss
            # S = true spectrum, S' = pred spectrum, D = design, D' = pred design
            # MSE(S, S') + MSE(D, D') + binary loss of each squre in pred design
            for_loss = output.flatten(start_dim=1)
            binary_loss = torch.mean(torch.square(for_loss * (for_loss - 1)))
            
            L_s = loss_fn(pred_spect, spect)
            # L_d = loss_fn(output, struct)
            L_b = binary_loss
            
            loss = weight_Ls * L_s + weight_Lb * L_b

            # scale loss by batch size
            running_loss[0] += L_s.item() * spect.shape[0]
            running_loss[1] += L_b.item() * spect.shape[0]

        # track loss over validation set
        running_loss[0] /= len(val_loader.dataset)
        running_loss[1] /= len(val_loader.dataset)
        val_loss_v_epoch.append(running_loss)
        print("Validation Loss, (L_s, L_b): {}".format(val_loss_v_epoch[epoch]))

except KeyboardInterrupt as e:
    print(e)

print('Training Completed in: {} secs'.format(time.time()-start))
######################################################################

# Save current parameters of inverse net
save_path = MODELS_PATH + "EM300InverseNN_E{}_Bstart{}_BIncr{}_Decay{}_Step{}_NewModel".format(epochs, bin_loss_start, incr_bin_loss, gamma, step_size)
save_path = MODELS_PATH + "EM300InverseNN2"
torch.save(inverse_model.state_dict(), save_path)
print("Inverse model trained and saved to: " + save_path)

save_f = "/scratch/gpfs/sbfisher/EM300/results/EMInverse2TrainLoss_BIncr{}.npy".format(bin_loss_start)
np.save(save_f, np.array(loss_v_epoch))
print("Loss results saved to: {}".format(save_f))
save_f = "/scratch/gpfs/sbfisher/EM300/results/EMInverse2ValidationLoss_BIncr{}.npy".format(bin_loss_start)
np.save(save_f, np.array(val_loss_v_epoch))
print("Val Loss results saved to: {}".format(save_f))

