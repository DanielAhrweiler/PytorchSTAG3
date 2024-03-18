import torch
import torch.nn as nn

#architecture for a regression NN, 1 hidden layer
class Regressor1(nn.Module):
	def __init__(self, inputSize, hiddenSizes, outputSize):
		super(Regressor1, self).__init__()
		self.hidden1 = nn.Linear(inputSize, hiddenSizes[0])
		self.output = nn.Linear(hiddenSizes[0], outputSize)
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()
		self.lrelu = nn.LeakyReLU(0.01)
		self.tanh = nn.Tanh()

	def forward(self, x, actFunctCode):
		hiddenOut1 = self.hidden1(x)
		#print(f'BEFORE AF | HL Weights = {self.hidden1.weight.tolist()}')
		#print(f'BEFORE AF | HL Biases = {self.hidden1.bias.tolist()}')
		#print(f'BEFORE AF | HL Outputs = {self.hidden1(x)}')
		if actFunctCode == 'SIGM':
			hiddenOut1 = self.sigmoid(hiddenOut1)
		elif actFunctCode == 'RELU':
			hiddenOut1 = self.relu(hiddenOut1)
		elif actFunctCode == 'LRELU':
			hiddenOut1 = self.lrelu(hiddenOut1)
		elif actFunctCode == 'TANH':
			hiddenOut1 = self.tanh(hiddenOut1)
		else:
			raise ValueError(f'Invalid activation code [{actFunctCode}]')
	
		#print(f'AFTER AF | HL Weights = {self.hidden1.weight.tolist()}')
		#print(f'AFTER AF | HL Biases = {self.hidden1.bias.tolist()}')
		output = self.output(hiddenOut1)
		return output

	def setLeakValue(self, leakVal):
		self.lrelu = nn.LeakyReLU(leakVal)


#architecture for a regression NN, 2 hidden layers
class Regressor2(nn.Module):
	def __init__(self, inputSize, hiddenSizes, outputSize):
		super(Regressor2, self).__init__()
		self.hidden1 = nn.Linear(inputSize, hiddenSizes[0])
		self.hidden2 = nn.Linear(hiddenSizes[0], hiddenSizes[1])
		self.output = nn.Linear(hiddenSizes[1], outputSize)#change back to hiddenSizes[0]?? 
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		hiddenOut1 = self.sigmoid(self.hidden1(x))
		hiddenOut2 = self.sigmoid(self.hidden2(hiddenOut1))
		output = self.output(hiddenOut2)
		return output

#architecture for a classification NN, 1 hidden layer
class Classifier1(nn.Module):
	def __init__(self, inputSize, hiddenSizes, outputSize):
		super(Classifier1, self).__init__()
		self.hidden1 = nn.Linear(inputSize, hiddenSizes[0])
		self.output = nn.Linear(hiddenSizes[0], outputSize)
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		hiddenOut1 = self.relu(self.hidden1(x))
		output = self.output(hiddenOut1)
		return self.softmax(output)

#architecture for a classification NN, 1 hidden layer
class Classifier2(nn.Module):
	def __init__(self, inputSize, hiddenSizes, outputSize):
		super(Classifier2, self).__init__()
		self.hidden1 = nn.Linear(inputSize, hiddenSizes[0])
		self.hidden2 = nn.Linear(hiddenSizes[0], hiddenSizes[1])
		self.output = nn.Linear(hiddenSizes[1], outputSize)
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
