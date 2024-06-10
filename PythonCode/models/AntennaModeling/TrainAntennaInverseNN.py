from AntennaInverseNetwork import AntennaInverseNN
from AntennaNetwork import AntennaCNN
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
class AntennaDataset(Dataset):
    def __init__(self, x_data_path, y_data_path):
        super().__init__()
        self.x = torch.tensor(np.load(x_data_path))
        self.y = torch.tensor(np.load(y_data_path))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

DATA_PATH = "./data/"
# Create datasets for training the inverse model. 
# Use the training dataset from forward network
x_train_path = DATA_PATH + "x_train.npy"
y_train_path = DATA_PATH + "y_train.npy"

x_test_path = DATA_PATH + "x_test.npy"
y_test_path = DATA_PATH + "y_test.npy"

batch_size = 512

train_dataset = AntennaDataset(x_train_path, y_train_path)
test_dataset = AntennaDataset(x_test_path, y_test_path)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size, shuffle=False)

# print(train_dataset[:][1].shape)
# print(test_dataset[:][0].shape)

# test dataset
# test_d = test_dataset[1]
# plt.imshow(test_dataset[1], cmap="gray")
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
device = "cpu"
print(f"Using {device} device")
input("enter to continue and start training")

MODELS_PATH = "./saved/"

forward_model = AntennaCNN()
# Load Forward network from saved state dict
forward_model.load_state_dict(torch.load(MODELS_PATH + "AntennaCNN"))
forward_model.eval()
forward_model = forward_model.to(device)

# inverse_model = AntennaInverseNN(m=20, use_threshold=True).to(device)
inverse_model = AntennaInverseNN().to(device)

######################################################################


# Train Inverse Model
######################################################################
print('Training....')
total = 0
correct = 0
start = time.time()

# Freeze Forward model params (maybe not necessary since the optimizer shouldn't change them???)
# for param in forward_model.parameters():
#     param.requires_grad = False

for name, child in forward_model.named_children():
    for _, param in child.named_parameters():
        param.requires_grad = False
    if "batchnorm" in name:
        child.running_mean.requires_grad = False
        child.running_var.requires_grad = False


loss_fn = nn.MSELoss()

# optimizer = torch.optim.Adam(inverse_model.parameters(), lr=0.001)#, betas=(1,1))
# try larger learning rate
optimizer = torch.optim.RAdam(inverse_model.parameters(), lr=0.01)

hyperparam_values = {"L_s": [1], "L_d": [0], "L_b": [10]}

# keep track of loss every epoch
losses_vs_params = {}
try:
    epochs = 100
    # Loop through all three hyperparams
    for m in range(len(hyperparam_values["L_s"])):
        for j in range(len(hyperparam_values["L_d"])):
            for k in range(len(hyperparam_values["L_b"])):
                weight_Ls = hyperparam_values["L_s"][m]
                weight_Ld = hyperparam_values["L_d"][j]
                weight_Lb = hyperparam_values["L_b"][k]
    
                first_loss = 0
                last_loss = 0
                loss_v_epoch = []
                grads_v_epoch = []
                hard_threshold = False 
                # Run training for this set of hyperparams
                for epoch in range(epochs):
                    if epoch == 100:
                        hard_threshold = True
                    
                    for i, data in enumerate(train_loader, 1):
                        struct, spect = data
                        struct = struct.to(device).float()
                        spect = spect.to(device).float()
                        print(struct[0])
                        print(spect[0])
                        
                        # if(torch.cuda.is_available()):
                        #     struct, result = labels.cuda()
                        optimizer.zero_grad()
                        
                        # spect should be batch of [n, 81] antenna s11 spectrums
                        # get batch outputs of inverse network
                        # set m based on epoch (start at 1 and increase a bit each epoch)
                        # if epoch >= 89: 
                        #     m = 20
                        # elif epoch >= 50:
                        #     m = (epoch/2)-24
                        # else:
                        #     m = 1
                        m=1
                        # training worked best so far with: m = 1
                        output = inverse_model(spect, m=m, use_threshold=True)
                        # print(output)
                        # try hard thresholding
                        if hard_threshold:
                            output = torch.where(output > 0.5, 1, 0)
                        
                        # now output is linear 144 layer, reshape to 12x12 structure
                        # reshape into input for predictor neural network
                        output = output.reshape([output.shape[0], 1, 12, 12])
                        
                        # get predicted spectrum from forward model using normalized output
                        pred_spect = forward_model(output) # normalize output
                        
                        # Compute loss
                        # S = true spectrum, S' = pred spectrum, D = design, D' = pred design
                        # MSE(S, S') + MSE(D, D') + binary loss of each squre in pred design
                        for_loss = output.flatten(start_dim=1)
                        # print(for_loss.shape)
                        binary_loss = torch.mean(torch.square(for_loss * (for_loss - 1)))
                        
                        L_s = loss_fn(pred_spect, spect)
                        L_d = loss_fn(output, struct)
                        L_b = binary_loss
    
                        loss = weight_Ls * L_s + weight_Ld * L_d + weight_Lb * L_b
                        loss_components = {"Ls": L_s, "Ld": L_d, "Lb": L_b}
                        
                        if(i%20 == 0):
                            print('Epoch: {} Batch: {} loss: {}'.format(epoch, i, loss.item()))
                            print("Loss Breakdown: Ls: {}, Ld: {}, Lb: {}".format(float(L_s), float(L_d), float(L_b)))
                            print("Current m: {}".format(m))
                            #print("Cumulative Correct/Total: {}/{}".format(correct, total))
                            # Save current parameters of inverse net
                            # torch.save(inverse_model.state_dict(), MODELS_PATH + "AntennaInverseNN")

                            # design created, middle row just to see values
                        print(output[0,0,5,:])
                        print(output.requires_grad)
                            
                        
                        # Backpropagate gradients
                        loss.backward()
                        for name, parameters in inverse_model.named_parameters():
                            print(name, parameters.shape)
                        print(inverse_model.network[4].weight)
                        optimizer.step()
                        break
    
                    # store loss after each epoch
                    loss_v_epoch.append((float(L_s), float(L_d), float(L_b)))
                    # Look at the gradients of the inverse model each epoch
                    # specifically last layer in this case since we are worried about it exploding
                    grads_v_epoch.append(output.cpu().detach())
                    
                    # store first epoch final loss and last epoch final loss
                    if (epoch == 0):
                        first_loss = list(loss_components)
                    elif (epoch == epochs-1):
                        last_loss = list(loss_components)
                        losses_vs_params[str((weight_Ls, weight_Ld, weight_Lb))] = (first_loss, last_loss)
                    break

except KeyboardInterrupt as e:
    print(e)

print('Training Completed in: {} secs'.format(time.time()-start))

# try:
#     # Store losses in file
#     with open("training_loses.json", 'w') as f:
#         json.dump(losses_vs_params, f)
# except Exception as e:
#     print(e)
# print('Training accuracy: {} %'.format((correct/total)*100))
######################################################################

# Save current parameters of inverse net
torch.save(inverse_model.state_dict(), MODELS_PATH + "AntennaInverseNN")
print("Inverse model trained and saved")

save_f = "/scratch/gpfs/sbfisher/TrainingResults/AntennaInverseLoss.npy"
save_f2 = "/scratch/gpfs/sbfisher/TrainingResults/AntennaInverseGrads.npy"

np.save(save_f, np.array(loss_v_epoch))
np.save(save_f2, np.array(grads_v_epoch))

print("Loss results saved to: {}".format(save_f))
print("Grad results saved to: {}".format(save_f2))

