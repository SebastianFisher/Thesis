import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn as nn
from EM300CNN import EM300CNN
from torch.utils.data import Dataset, DataLoader
import time

RANDOM_SEED = 10
torch.manual_seed(RANDOM_SEED)
##################################################################
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

# Create dataloader
# batch_size = 512
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

# Load pretrained model
model = EM300CNN(mean=0.4068).to(device)
print(model.load_state_dict(torch.load(MODELS_PATH + "EM300CNN")))

##################################################################
# TRAINING 
print('Training....')
start = time.time()

loss_fn = nn.L1Loss()

# lrs = [0.0005, 0.001, 0.005, 0.01, 0.03, 0.05]
lrs = [0.001]
epochs = 70
# next try, epochs=150, with gamma 0.92, every 5 epochs decrease?

losses_arrs = []
val_losses_arrs = []

# EDIT this back in for real training loop
gamma = 0.8
step_size=5

try:
    for lr in lrs:
        print("Training with LR: {}, epochs: {}".format(lr, epochs))
        print("Training with scheduler, {} * lr every 5 epochs".format(gamma))
        print("Starting with no tanh activation at output")
        # load pretrained matlab weights
        model.load_state_dict(torch.load(MODELS_PATH + "EM300CNN"))
        
        # try various learning rates
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # decay learning rate by gamma every 5 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
        losses = np.zeros(epochs)
        val_losses = np.zeros(epochs)
        # Train model on new data
        for epoch in tqdm(range(epochs)):
            model.train()
            # running_loss = 0
            for i, data in enumerate(train_loader, 1):
                struct, spect = data
                struct = struct.to(device).float()
                spect = spect.to(device).float()
                
                optimizer.zero_grad()
                
                # model forward pass
                predicted_spectrum = model(struct, apply_tanh=True)
                # predicted_spectrum = model(struct, apply_tanh=False)
                
                # For loss, use MSE between predicted spectrum and actual spectrum
                loss = loss_fn(spect, predicted_spectrum)
        
                if(i%20 == 0):
                    print('Epoch: {} Batch: {} loss: {}'.format(epoch, i, loss.item()))
                    print("Loss: " + str(loss.item()))
        
                # Backpropagate gradients, optimizer step
                loss.backward()
                optimizer.step()
    
                # add loss scaled by number of items in this batch
                # running_loss += loss.item() * struct.shape[0]
    
            model.eval()
            running_loss = 0
            # test on training set
            for data in train_loader:
                struct, spect = data
                struct = struct.to(device).float()
                spect = spect.to(device).float()
                
                # model forward pass
                predicted_spectrum = model(struct, apply_tanh=True)
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
                predicted_spectrum = model(struct, apply_tanh=True)
                # predicted_spectrum = model(struct, apply_tanh=False)
                
                # For loss, use MSE between predicted spectrum and actual spectrum
                loss = loss_fn(spect, predicted_spectrum)
    
                # scale loss by batch size
                running_loss += loss.item() * spect.shape[0]
    
            # track loss over validation set
            val_losses[epoch] = running_loss / len(val_loader.dataset)
            print("Validation Loss: {}".format(val_losses[epoch]))
    
            # increase learning rate every 5 epochs
            scheduler.step() # EDIT this back in for real training
        
        # Save current parameters of inverse net
        torch.save(model.state_dict(), MODELS_PATH + "EM300CNNTunedLR{}_gamma{}_Every{}_E{}_Bs_{}".format(lr, gamma, step_size, epochs, batch_size))
        print("EM300CNN forward model trained and saved for lr={}, decaying {} every 5 epochs, {} epochs".format(lr, gamma, epochs))
        
        # torch.save(model.state_dict(), MODELS_PATH + "EM300CNNTunedLR{}".format(lr))
        # print("EM300CNN forward model trained and saved for lr={}, 20 epochs".format(lr))
        
        losses_arrs.append(losses)
        val_losses_arrs.append(val_losses)

except KeyboardInterrupt as e:
    print(e)

print('Training Completed in: {} secs'.format(time.time()-start))
loss_filename = "/scratch/gpfs/sbfisher/EM300/results/EM300CNNTrainingLoss.npy"
print("Saving losses to: {}".format(loss_filename))
np.save(loss_filename,  np.array(losses_arrs))
val_loss_filename = "/scratch/gpfs/sbfisher/EM300/results/EM300CNNValidationLoss.npy"
print("Saving validation losses to: {}".format(val_loss_filename))
np.save(val_loss_filename, np.array(val_losses_arrs))
######################################################################

