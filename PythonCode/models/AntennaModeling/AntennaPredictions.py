from AntennaNetwork import AntennaCNN



model = AntennaCNN()


for name, param in model.named_parameters():
    print(name, param.size())

