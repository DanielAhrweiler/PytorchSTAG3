import torch
import torch.nn as nn

#define the neural network architecture
class H1NN(nn.Module):
	def __init__(self, inputSize, hiddenSize1, outputSize):
		super(H1NN, self).__init__()
		self.hidden1 = nn.Linear(inputSize, hiddenSize1)
		self.sigmoid = nn.Sigmoid()
		self.output = nn.Linear(hiddenSize1, outputSize)

	def forward(self, x):
		hiddenOut1 = self.sigmoid(self.hidden1(x))
		output = self.output(hiddenOut1)
		return output

#define the neural network architecture
class H2NN(nn.Module):
	def __init__(self, inputSize, hiddenSize1, hiddenSize2, outputSize):
		super(H2NN, self).__init__()
		self.hidden1 = nn.Linear(inputSize, hiddenSize1)
		self.hidden2 = nn.Linear(hiddenSize1, hiddenSize2)
		self.sigmoid = nn.Sigmoid()
		self.output = nn.Linear(hiddenSize1, outputSize)

	def forward(self, x):
		hiddenOut1 = self.sigmoid(self.hidden1(x))
		hiddenOut2 = self.sigmoid(self.hidden2(hiddenOut1))
		output = self.output(hiddenOut1)
		return output
