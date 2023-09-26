import torch
import torch.nn as nn

#architecture for a regression NN, 1 hidden layer
class Regressor1(nn.Module):
	def __init__(self, inputSize, hiddenSize1, outputSize):
		super(Regressor1, self).__init__()
		self.hidden1 = nn.Linear(inputSize, hiddenSize1)
		self.output = nn.Linear(hiddenSize1, outputSize)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		hiddenOut1 = self.sigmoid(self.hidden1(x))
		output = self.output(hiddenOut1)
		return output

#architecture for a regression NN, 2 hidden layers
class Regressor2(nn.Module):
	def __init__(self, inputSize, hiddenSize1, hiddenSize2, outputSize):
		super(Regressor2, self).__init__()
		self.hidden1 = nn.Linear(inputSize, hiddenSize1)
		self.hidden2 = nn.Linear(hiddenSize1, hiddenSize2)
		self.output = nn.Linear(hiddenSize1, outputSize)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		hiddenOut1 = self.sigmoid(self.hidden1(x))
		hiddenOut2 = self.sigmoid(self.hidden2(hiddenOut1))
		output = self.output(hiddenOut2)
		return output

#architecture for a classification NN, 1 hidden layer
class Classifier1(nn.Module):
	def __init__(self, inputSize, hiddenSize, outputSize):
		super(Classifier1, self).__init__()
		self.hidden1 = nn.Linear(inputSize, hiddenSize)
		self.output = nn.Linear(hiddenSize, outputSize)
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		hiddenOut1 = self.relu(self.hidden1(x))
		output = self.output(hiddenOut1)
		return self.softmax(output)

#architecture for a classification NN, 1 hidden layer
class Classifier2(nn.Module):
	def __init__(self, inputSize, hiddenSize1, hiddenSize2, outputSize):
		super(Classifier2, self).__init__()
		self.hidden1 = nn.Linear(inputSize, hiddenSize1)
		self.hidden2 = nn.Linear(hiddenSize1, hiddenSize2)
		self.output = nn.Linear(hiddenSize2, outputSize)
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		hiddenOut1 = self.relu(self.hidden1(x))
		hiddenOut2 = self.relu(self.hidden2(hiddenOut1))
		output = self.output(hiddenOut2)
		return self.softmax(output)

#architecture for a classification NN, unknown hidden layers (at least one)
class ClassifierX(nn.Module):
	def __init__(self, inputSize, hiddenSizes, outputSize):
		super(ClassifierX, self).__init__()
		self.hiddens = []
		for i in range(len(hiddenSizes)):
			if i == 0:
				self.hiddens.append(nn.Linear(inputSize, hiddenSizes[0]))
			else:
				self.hiddens.append(nn.Linear(hiddenSizes[i-1], hiddenSizes[i]))
		self.output = nn.Linear(hiddenSizes[-1], outputSize)
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		for i in range(len(self.hiddens)):
			if i == 0:
				itrHiddenOut = self.relu(self.hiddens[i](x))
			else:
				itrHiddenOut = self.relu(self.hiddens[i](itrHiddenOut))
		output = self.output(itrHiddenOut)
		return self.softmax(output)
