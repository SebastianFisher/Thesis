import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn as nn
from EM300CNN import EM300CNN    # CNN architecture is the exact same as 300 micron
from torch.utils.data import Dataset, DataLoader
import time
import os

RANDOM_SEED = 10
torch.manual_seed(RANDOM_SEED)
##################################################################
# Dataset object for DataLoader
class EM200Dataset(Dataset):
    def __init__(self, x_data_path, y_data_path):
        super().__init__()
        self.x = torch.tensor(np.load(x_data_path))
        self.y = torch.tensor(np.load(y_data_path))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].unsqueeze(0), self.y[idx]

# Load datasets
DATA_DIR = "/scratch/gpfs/sbfisher/EM200/data/"
# Create datasets for training the inverse model. 
# Use the training dataset from forward network
x_data_path = DATA_DIR + "x_data.npy"
y_data_path = DATA_DIR + "y_data.npy"

dataset = EM200Dataset(x_data_path, y_data_path)

# shortened_dataset = dataset[:len(dataset)//4]
# print(len(shortened_dataset))
# only use 25% of available dataset (~ 12.5k structure, x6, 75k total)
generator = torch.Generator().manual_seed(RANDOM_SEED)
# train, val, test, held_out = torch.utils.data.random_split(dataset, [0.2, 0.025, 0.025, 0.75], generator=generator)
train, val, test = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

# Create dataloader
batch_size = 4096

train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size, shuffle=True, generator=generator)
val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test,
                                           batch_size=batch_size, shuffle=False)

##################################################################
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

MODELS_PATH = "./saved/"
PRETRAINED_MODELS_PATH = "/home/sbfisher/Cadence/Thesis/PythonCode/models/EM300Modeling/saved/"

# PRETRAINED_MODEL = "EM300CNNTunedLR0.001_gamma0.8_Every5_E50_Bs_4096"
PRETRAINED_MODEL = "EM300CNN" # try matlab 300 model
# Load pretrained model, for this we transfer weights from 
# theoretical mean of input should be 0.4074 = (16 * 16 * 0.5 + 4) / (18 * 18) from
# the way that structures are generated, but this is the batch mean that I think Emir's program
# calculated. Either way it should be close enough to mean zero...
model = EM300CNN(mean=0.4068).to(device)
print(model.load_state_dict(torch.load(PRETRAINED_MODELS_PATH + PRETRAINED_MODEL)))

##################################################################
# TRAINING 
print('Training....')
start = time.time()

loss_fn = nn.L1Loss()

lrs = [0.001]
# lrs = [0.005, 0.01, 0.03, 0.05]
epochs = 40
losses_arrs = []
val_losses_arrs = []

gamma = 0.8

print("Training set size: ", len(train_loader.dataset))
try:
    for lr in lrs:
        print("Training with LR: {}, epochs: {}".format(lr, epochs))
        print("Training with scheduler, 0.8 * lr every 4 epochs")
        # load pretrained matlab weights
        model.load_state_dict(torch.load(PRETRAINED_MODELS_PATH + PRETRAINED_MODEL))
        
        # try various learning rates
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # decay learning rate by 0.8 every 4 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=gamma)

        losses = np.zeros(epochs)
        val_losses = np.zeros(epochs)
        # Train model on new data
        for epoch in tqdm(range(epochs)):
            model.train()
            running_loss = 0
            for i, data in enumerate(train_loader, 1):
                struct, spect = data
                struct = struct.to(device).float()
                spect = spect.to(device).float()
                
                optimizer.zero_grad()
                
                # model forward pass
                predicted_spectrum = model(struct)
                # predicted_spectrum = model(struct, apply_tanh=False)
                
                # For loss, use MSE between predicted spectrum and actual spectrum
                loss = loss_fn(spect, predicted_spectrum)
        
                if(i%20 == 0):
                    print('Epoch: {} Batch: {} loss: {}'.format(epoch, i, loss.item()))
                    print("Loss: " + str(loss.item()))
        
                # Backpropagate gradients, optimizer step
                loss.backward()
                optimizer.step()

            print("Testing on training set")
            model.eval()
            running_loss = 0
            # test on training set
            for data in train_loader:
                struct, spect = data
                struct = struct.to(device).float()
                spect = spect.to(device).float()
                
                # model forward pass
                predicted_spectrum = model(struct)
                # predicted_spectrum = model(struct, apply_tanh=False)
                
                # For loss, use MSE between predicted spectrum and actual spectrum
                loss = loss_fn(spect, predicted_spectrum)
    
                # scale loss by batch size
                running_loss += loss.item() * spect.shape[0]
            
            # Calculate training loss for this epoch and store it
            losses[epoch] = running_loss / len(train_loader.dataset)
            print("Total Training Loss: {}".format(losses[epoch]))
            
            print("Testing on validation set")
            # test on validation set
            model.eval()
            running_loss = 0
            for data in val_loader:
                struct, spect = data
                struct = struct.to(device).float()
                spect = spect.to(device).float()
                
                # model forward pass
                predicted_spectrum = model(struct)
                # predicted_spectrum = model(struct, apply_tanh=False)
                
                # For loss, use MSE between predicted spectrum and actual spectrum
                loss = loss_fn(spect, predicted_spectrum)

                # scale loss by batch size
                running_loss += loss.item() * spect.shape[0]
    
            # track loss over validation set
            val_losses[epoch] = running_loss / len(val_loader.dataset)
            print("Validation Loss: {}".format(val_losses[epoch]))

            # increase learning rate every 4 epochs
            scheduler.step()
        
        # Save current parameters of inverse net
        save_name = MODELS_PATH + "EM200CNNTunedDecayingLR{}_By{}_every4_Bs{}_E{}".format(lr, gamma, batch_size, epochs)
        torch.save(model.state_dict(), save_name)
        print("EM200CNN forward model trained and saved for lr={}, decaying by {} every 4 epochs".format(lr, gamma))
        print("Saved to: " + save_name)
        
        losses_arrs.append(losses)
        val_losses_arrs.append(val_losses)

except KeyboardInterrupt as e:
    print(e)

LOSS_DIR = "/scratch/gpfs/sbfisher/EM200/results/"
if not os.path.exists(LOSS_DIR):
    os.makedirs(LOSS_DIR)

print('Training Completed in: {} secs'.format(time.time()-start))
loss_filename = "/scratch/gpfs/sbfisher/EM200/results/EM200CNNTrainingLoss.npy"
print("Saving losses to: {}".format(loss_filename))
np.save(loss_filename,  np.array(losses_arrs))
val_loss_filename = "/scratch/gpfs/sbfisher/EM200/results/EM200CNNValidationLoss.npy"
print("Saving validation losses to: {}".format(val_loss_filename))
np.save(val_loss_filename, np.array(val_losses_arrs))
######################################################################



