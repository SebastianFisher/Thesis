import torch
import torch.nn as nn

class InputNormalize(nn.Module):
    def __init__(self, mean):
        super(InputNormalize, self).__init__()
        self.mean = mean

    def forward(self, x):
        x -= self.mean
        
        return x

class EM300CNN(nn.Module):
    def __init__(self, mean=0.4068):
        super(EM300CNN, self).__init__()

        # self.normalize = InputNormalize(mean)
        self.mean = mean

        # Conv Layer 1
        """
        'conv_1'         2-D Convolution       64 12×12×1 convolutions with stride [1  1] and padding 'same'
        Padding 'same' equals the padding needed to keep the output the same size
        'batchnorm_1'    Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_1'    Leaky ReLU            Leaky ReLU with scale 0.01_
        """
        self.conv_1 = nn.Conv2d(1, 64, kernel_size=12, stride=1, padding='same')
        self.batchnorm_1 = nn.BatchNorm2d(64)
        self.leakyrelu_1 = nn.LeakyReLU(0.01)

        # Conv Layer 2
        """
        'conv_2'         2-D Convolution       64 10×10×64 convolutions with stride [1  1] and padding 'same'
        'batchnorm_2'    Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_2'    Leaky ReLU            Leaky ReLU with scale 0.01_
        """
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=10, stride=1, padding='same')
        self.batchnorm_2 = nn.BatchNorm2d(64)
        self.leakyrelu_2 = nn.LeakyReLU(0.01)

        # Conv Layer 3
        """
        'conv_3'         2-D Convolution       64 8×8×64 convolutions with stride [1  1] and padding 'same'
        'batchnorm_3'    Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_3'    Leaky ReLU            Leaky ReLU with scale 0.01
        """
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding='same')
        self.batchnorm_3 = nn.BatchNorm2d(64)
        self.leakyrelu_3 = nn.LeakyReLU(0.01)

        # Conv Layer 4
        """
        'conv_4'         2-D Convolution       64 6×6×64 convolutions with stride [1  1] and padding 'same'
        'batchnorm_4'    Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_4'    Leaky ReLU            Leaky ReLU with scale 0.01_
        """
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=6, stride=1, padding='same')
        self.batchnorm_4 = nn.BatchNorm2d(64)
        self.leakyrelu_4 = nn.LeakyReLU(0.01)

        """
         'conv_5'         2-D Convolution       64 5×5×64 convolutions with stride [1  1] and padding 'same'
         'batchnorm_5'    Batch Normalization   Batch normalization with 64 channels
         'leakyrelu_5'    Leaky ReLU            Leaky ReLU with scale 0.01
        """
        self.conv_5 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same')
        self.batchnorm_5 = nn.BatchNorm2d(64)
        self.leakyrelu_5 = nn.LeakyReLU(0.01)

        """
        'conv_6'         2-D Convolution       64 5×5×64 convolutions with stride [1  1] and padding 'same'
        'batchnorm_6'    Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_6'    Leaky ReLU            Leaky ReLU with scale 0.01
        """
        self.conv_6 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same')
        self.batchnorm_6 = nn.BatchNorm2d(64)
        self.leakyrelu_6 = nn.LeakyReLU(0.01)

        """
        'conv_7'         2-D Convolution       64 4×4×64 convolutions with stride [1  1] and padding 'same'
        'batchnorm_7'    Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_7'    Leaky ReLU            Leaky ReLU with scale 0.01
        """
        self.conv_7 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding='same')
        self.batchnorm_7 = nn.BatchNorm2d(64)
        self.leakyrelu_7 = nn.LeakyReLU(0.01)

        """
        'conv_8'         2-D Convolution       64 4×4×64 convolutions with stride [1  1] and padding 'same'
        'batchnorm_8'    Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_8'    Leaky ReLU            Leaky ReLU with scale 0.01
        """
        self.conv_8 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding='same')
        self.batchnorm_8 = nn.BatchNorm2d(64)
        self.leakyrelu_8 = nn.LeakyReLU(0.01)

        """
        'conv_9'         2-D Convolution       64 4×4×64 convolutions with stride [1  1] and padding 'same'
        'batchnorm_9'    Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_9'    Leaky ReLU            Leaky ReLU with scale 0.01
        """
        self.conv_9 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding='same')
        self.batchnorm_9 = nn.BatchNorm2d(64)
        self.leakyrelu_9 = nn.LeakyReLU(0.01)

        """
        'conv_10'        2-D Convolution       64 3×3×64 convolutions with stride [1  1] and padding 'same'
        'batchnorm_10'   Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_10'   Leaky ReLU            Leaky ReLU with scale 0.01
        """
        self.conv_10 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        self.batchnorm_10 = nn.BatchNorm2d(64)
        self.leakyrelu_10 = nn.LeakyReLU(0.01)

        """
        'conv_11'        2-D Convolution       64 3×3×64 convolutions with stride [1  1] and padding 'same'
        'batchnorm_11'   Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_11'   Leaky ReLU            Leaky ReLU with scale 0.01
        """
        self.conv_11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        self.batchnorm_11 = nn.BatchNorm2d(64)
        self.leakyrelu_11 = nn.LeakyReLU(0.01)

        """
        'conv_12'        2-D Convolution       64 3×3×64 convolutions with stride [1  1] and padding 'same'
        'batchnorm_12'   Batch Normalization   Batch normalization with 64 channels
        'leakyrelu_12'   Leaky ReLU            Leaky ReLU with scale 0.01
        """
        self.conv_12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        self.batchnorm_12 = nn.BatchNorm2d(64)
        self.leakyrelu_12 = nn.LeakyReLU(0.01)

        # 'dropout_1'      Dropout               50% dropout
        self.dropout_1 = nn.Dropout(0.5)

        """
        'fc_1'           Fully Connected       500 fully connected layer
        'batchnorm_13'   Batch Normalization   Batch normalization with 500 channels
        'leakyrelu_13'   Leaky ReLU            Leaky ReLU with scale 0.01
        """
        # output of previous conv2d layer is 64 filters 18x18 still
        self.fc_1 = nn.Linear(64 * 18 * 18, 500)
        self.batchnorm_13 = nn.BatchNorm1d(500)
        self.leakyrelu_13 = nn.LeakyReLU(0.01)

        # 'dropout_2'      Dropout               50% dropout
        self.dropout_2 = nn.Dropout(0.5)

        """
        'fc_2'           Fully Connected       500 fully connected layer
        'batchnorm_14'   Batch Normalization   Batch normalization with 500 channels
        'leakyrelu_14'   Leaky ReLU            Leaky ReLU with scale 0.01
        """
        self.fc_2 = nn.Linear(500, 500)
        self.batchnorm_14 = nn.BatchNorm1d(500)
        self.leakyrelu_14 = nn.LeakyReLU(0.01)

        # 'dropout_3'      Dropout               50% dropout
        self.dropout_3 = nn.Dropout(0.5)

        """
        'fc_3'           Fully Connected       500 fully connected layer
        'batchnorm_15'   Batch Normalization   Batch normalization with 500 channels
        'leakyrelu_15'   Leaky ReLU            Leaky ReLU with scale 0.01
        """
        self.fc_3 = nn.Linear(500, 500)
        self.batchnorm_15 = nn.BatchNorm1d(500)
        self.leakyrelu_15 = nn.LeakyReLU(0.01)

        # 'dropout_4'      Dropout               50% dropout
        self.dropout_4 = nn.Dropout(0.5)

        """
        'fc_4'           Fully Connected       500 fully connected layer
        'batchnorm_16'   Batch Normalization   Batch normalization with 500 channels
        'leakyrelu_16'   Leaky ReLU            Leaky ReLU with scale 0.01
        """
        self.fc_4 = nn.Linear(500, 500)
        self.batchnorm_16 = nn.BatchNorm1d(500)
        self.leakyrelu_16 = nn.LeakyReLU(0.01)


        # 'dropout_5'      Dropout               50% dropout
        self.dropout_5 = nn.Dropout(0.5)

        """
        'fc_5'           Fully Connected       54 fully connected layer
        'tanh'           Tanh                  Hyperbolic tangent
        'mae'            Regression Output     Mean absolute error
        """ 
        self.fc_5 = nn.Linear(500, 54)
        self.tanh = nn.Tanh()



    def forward(self, x):
        x = x - self.mean
        
        # x = self.normalize(x)
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

        # Need to do reshaping since the weights trained from matlab assume
        # flattening in column major order, but this will flatten in row-major
        # order. So we simply swap the rows and columns before flattening
        x = x.permute(0, 1, 3, 2)
        x = x.flatten(start_dim=1) # to match linear input shape

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

if __name__ == "__main__":
    # Create an instance of the EM300CNN model
    model = EM300CNN()
    
    # Print the weights
    for name, params in model.named_parameters():
        print(name, params.size())


