import torch.nn as nn
import torch.nn.functional as F


class VanillaCNN2(nn.Module):
    print("Loading VANILLA 2...")

    def __init__(self,  num_classes=10, input_channels=3):
        super(VanillaCNN2, self).__init__()  # (3,32,32)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2)

        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2)
        self.conv6 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2)

        self.conv7 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=2)
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2)

        self.pool = nn.MaxPool2d(2, 2)

        self.linear = nn.Linear(6400, 1024)
        self.linear1 = nn.Linear(1024, 128)
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self, x):  # (3,32,32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # print(x.size())

        x = self.linear(x)
        x = self.linear1(x)
        output = self.linear2(x)

        return output
