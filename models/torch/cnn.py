import torch
import torch.nn as nn

class Conv1d(nn.Module):

    def __init__(self):
        super().__init__(feature_size, hidden_size, output_size)
        self.conv1d = nn.Conv1d()
        self.maxpool = nn.MaxPool1d()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(conv_output, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)



    def forward():
        

        return
    
    