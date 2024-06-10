# THIS NOTEBOOK IS FOR TESTING THE TRAINED EM300CNN model
from EM300CNN import EM300CNN
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm

# RUN THIS TO LOAD THE MODEL AND TEST IT
import matplotlib.pyplot as plt

RANDOM_SEED = 10

# which device?
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# device = "cpu"
print(f"Using {device} device")

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
DATA_DIR = "/scratch/gpfs/sbfisher/EM250x300/data/"
# Create datasets for training the inverse model. 
# Use the training dataset from forward network
x_data_path = DATA_DIR + "x_data.npy"
y_data_path = DATA_DIR + "y_data.npy"

dataset = EM200Dataset(x_data_path, y_data_path)

generator = torch.Generator().manual_seed(RANDOM_SEED)
train, val, test = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator=generator)
batch_size = 1024
# val_loader = DataLoader(val, batch_size=512, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)


MODELS_PATH = "./saved/"
# Load pretrained model from 300x300 network data
model = EM300CNN(mean=0.4068).to(device)
PRETRAINED_MODEL_PATH = MODELS_PATH + "EM250x300CNNTunedDecayingLR0.0003_By0.8_every4"
model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
model.eval()

# Run training for this set of hyperparams
print("Running Tuned EM300CNN on 250x300 Test set")
tot_loss = 0
loss_fn = torch.nn.L1Loss()

for i, data in tqdm(enumerate(test_loader, 1)):
    struct, spect = data
    struct = struct.to(device).float()
    spect = spect.to(device).float()
    
    # get predicted spect of struct
    output = model(struct)
    
    loss = loss_fn(spect, output)
    tot_loss += loss.item() * struct.shape[0]

tot_loss /= len(test_loader.dataset)
print("Test L1 loss: {}".format(tot_loss))
    

