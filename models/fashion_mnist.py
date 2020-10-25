import torch
import torch.nn as nn
from utils.spectralpool import HartleyPool2d, CosinePool2d

class FashionCNN(nn.Module):
	def __init__(self):
		super(FashionCNN, self).__init__()

		self.layer1 = nn.Sequential(
							nn.Conv2d(1, 32, kernel_size=3, padding=1), 
							nn.BatchNorm2d(32), 
							nn.ReLU(), 
							nn.MaxPool2d(2)
						)
		self.layer2 = nn.Sequential(
							nn.Conv2d(32, 64, kernel_size=3), 
							nn.BatchNorm2d(64), 
							nn.ReLU(), 
							nn.MaxPool2d(2)
						)
		self.fc1 = nn.Linear(64*6*6, 600)
		self.drop = nn.Dropout2d(0.25)
		self.fc2 = nn.Linear(600, 120)
		self.fc3 = nn.Linear(120, 10)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.contiguous().view(out.size(0), -1)
		out = self.fc1(out)
		out = self.drop(out)
		out = self.fc2(out)
		out = self.fc3(out)

		return out

class HartleyPoolFashionCNN(nn.Module):
	def __init__(self):
		super(HartleyPoolFashionCNN, self).__init__()

		self.layer1 = nn.Sequential(
							nn.Conv2d(1, 32, kernel_size=3, padding=1), 
							nn.BatchNorm2d(32), 
							nn.ReLU(), 
							HartleyPool2d(14)
						)
		self.layer2 = nn.Sequential(
							nn.Conv2d(32, 64, kernel_size=3), 
							nn.BatchNorm2d(64), 
							nn.ReLU(), 
							HartleyPool2d(7)
						)
		self.fc1 = nn.Linear(64*6*6, 600)
		self.drop = nn.Dropout2d(0.25)
		self.fc2 = nn.Linear(600, 120)
		self.fc3 = nn.Linear(120, 10)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.contiguous().view(out.size(0), -1)
		out = self.fc1(out)
		out = self.drop(out)
		out = self.fc2(out)
		out = self.fc3(out)
		
		return out

class CosinePoolFashionCNN(nn.Module):
	def __init__(self):
		super(CosinePoolFashionCNN, self).__init__()

		self.layer1 = nn.Sequential(
							nn.Conv2d(1, 32, kernel_size=3, padding=1), 
							nn.BatchNorm2d(32), 
							nn.ReLU(), 
							CosinePool2d(14)
						)
		self.layer2 = nn.Sequential(
							nn.Conv2d(32, 64, kernel_size=3), 
							nn.BatchNorm2d(64), 
							nn.ReLU(), 
							CosinePool2d(6)
						)
		self.fc1 = nn.Linear(64*6*6, 600)
		self.drop = nn.Dropout2d(0.25)
		self.fc2 = nn.Linear(600, 120)
		self.fc3 = nn.Linear(120, 10)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.contiguous().view(out.size(0), -1)
		out = self.fc1(out)
		out = self.drop(out)
		out = self.fc2(out)
		out = self.fc3(out)

		return out