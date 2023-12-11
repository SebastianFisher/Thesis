import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Conv Layer 1
        """
        'conv_1'         2-D Convolution       64 12×12×1 convolutions with stride [1  1] and padding 'same'
        'batchnorm_1'    Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_1'    Leaky ReLU            Leaky ReLU with scale 0.01_
        """
        self.conv_1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(64)
        self.leakyrelu_1 = nn.LeakyReLU(0.01)

        # Conv Layer 2
        """
        'conv_2'         2-D Convolution       64 10×10×64 convolutions with stride [1  1] and padding 'same'
        'batchnorm_2'    Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_2'    Leaky ReLU            Leaky ReLU with scale 0.01_
        """
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(64)
        self.leakyrelu_2 = nn.LeakyReLU(0.01)

        # Conv Layer 3
        """
        'conv_3'         2-D Convolution       64 8×8×64 convolutions with stride [1  1] and padding 'same'
        'batchnorm_3'    Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_3'    Leaky ReLU            Leaky ReLU with scale 0.01
        """
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_3 = nn.BatchNorm2d(64)
        self.leakyrelu_3 = nn.LeakyReLU(0.01)

        # Conv Layer 4
        """
        'conv_4'         2-D Convolution       64 6×6×64 convolutions with stride [1  1] and padding 'same'
        'batchnorm_4'    Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_4'    Leaky ReLU            Leaky ReLU with scale 0.01_
        """
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_4 = nn.BatchNorm2d(64)
        self.leakyrelu_4 = nn.LeakyReLU(0.01)

        
        self.conv_5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_5 = nn.BatchNorm2d(64)
        self.leakyrelu_5 = nn.LeakyReLU(0.01)

        self.conv_6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_6 = nn.BatchNorm2d(64)
        self.leakyrelu_6 = nn.LeakyReLU(0.01)

        self.conv_7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_7 = nn.BatchNorm2d(64)
        self.leakyrelu_7 = nn.LeakyReLU(0.01)

        self.conv_8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_8 = nn.BatchNorm2d(64)
        self.leakyrelu_8 = nn.LeakyReLU(0.01)

        self.conv_9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_9 = nn.BatchNorm2d(64)
        self.leakyrelu_9 = nn.LeakyReLU(0.01)

        self.conv_10 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_10 = nn.BatchNorm2d(64)
        self.leakyrelu_10 = nn.LeakyReLU(0.01)

        self.conv_11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_11 = nn.BatchNorm2d(64)
        self.leakyrelu_11 = nn.LeakyReLU(0.01)

        self.conv_12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_12 = nn.BatchNorm2d(64)
        self.leakyrelu_12 = nn.LeakyReLU(0.01)

        self.dropout_1 = nn.Dropout(0.5)

        self.fc_1 = nn.Linear(64 * 18 * 18, 500)
        self.batchnorm_13 = nn.BatchNorm1d(500)
        self.leakyrelu_13 = nn.LeakyReLU(0.01)

        self.dropout_2 = nn.Dropout(0.5)

        self.fc_2 = nn.Linear(500, 500)
        self.batchnorm_14 = nn.BatchNorm1d(500)
        self.leakyrelu_14 = nn.LeakyReLU(0.01)

        self.dropout_3 = nn.Dropout(0.5)

        self.fc_3 = nn.Linear(500, 500)
        self.batchnorm_15 = nn.BatchNorm1d(500)
        self.leakyrelu_15 = nn.LeakyReLU(0.01)

        self.dropout_4 = nn.Dropout(0.5)

        self.fc_4 = nn.Linear(500, 500)
        self.batchnorm_16 = nn.BatchNorm1d(500)
        self.leakyrelu_16 = nn.LeakyReLU(0.01)

        self.dropout_5 = nn.Dropout(0.5)

        self.fc_5 = nn.Linear(500, 54)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batchnorm_1(x)
        x = self.leakyrelu_1(x)

        x = self.conv_2(x)
        x = self.batchnorm_2(x)
        x = self.leakyrelu_2(x)

        x = self.conv_3(x)
        x = self.batchnorm_3(x)
        x = self.leakyrelu_3(x)

        x = self.conv_4(x)
        x = self.batchnorm_4(x)
        x = self.leakyrelu_4(x)

        x = self.conv_5(x)
        x = self.batchnorm_5(x)
        x = self.leakyrelu_5(x)

        x = self.conv_6(x)
        x = self.batchnorm_6(x)
        x = self.leakyrelu_6(x)

        x = self.conv_7(x)
        x = self.batchnorm_7(x)
        x = self.leakyrelu_7(x)

        x = self.conv_8(x)
        x = self.batchnorm_8(x)
        x = self.leakyrelu_8(x)

        x = self.conv_9(x)
        x = self.batchnorm_9(x)
        x = self.leakyrelu_9(x)

        x = self.conv_10(x)
        x = self.batchnorm_10(x)
        x = self.leakyrelu_10(x)

        x = self.conv_11(x)
        x = self.batchnorm_11(x)
        x = self.leakyrelu_11(x)

        x = self.conv_12(x)
        x = self.batchnorm_12(x)
        x = self.leakyrelu_12(x)

        x = self.dropout_1(x)

        x = x.view(-1, 64 * 18 * 18)

        x = self.fc_1(x)
        x = self.batchnorm_13(x)
        x = self.leakyrelu_13(x)

        x = self.dropout_2(x)

        x = self.fc_2(x)
        x = self.batchnorm_14(x)
        x = self.leakyrelu_14(x)

        x = self.dropout_3(x)

        x = self.fc_3(x)
        x = self.batchnorm_15(x)
        x = self.leakyrelu_15(x)

        x = self.dropout_4(x)

        x = self.fc_4(x)
        x = self.batchnorm_16(x)
        x = self.leakyrelu_16(x)

        x = self.dropout_5(x)

        x = self.fc_5(x)
        x = self.tanh(x)

        return x

# Create an instance of the CustomCNN model
custom_cnn = CustomCNN()

# Print the architecture
print(custom_cnn)
