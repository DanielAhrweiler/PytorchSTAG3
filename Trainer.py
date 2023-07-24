import os
import math
from datetime import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
import StagNN
import AhrUtil as au

#======= Functions =======


#reads in data from file and returns Torch Tensors (feat space and target)
def fileToTensors(fpath, batchSize):
	featureSpace = []
	target = []
	fsBatch = []
	tBatch = []
	with open(fpath, 'r') as itrFile:
		for itrLine in itrFile:
			data = itrLine.strip().split(',')
			fsBatch.append(list(map(float, data[:(len(data)-1)])))
			tBatch.append([float(data[len(data)-1])])
			if (len(tBatch) >= batchSize):
				featureSpace.append(fsBatch)
				target.append(tBatch)
				fsBatch = []
				tBatch = []
	if (len(featureSpace) == 0):
		featureSpace.append(fsBatch)
		target.append(tBatch)
	fsTensor = torch.tensor(featureSpace)
	targetTensor = torch.tensor(target)
	return fsTensor, targetTensor


#======= Data Management =======

#define vars for custom db
startDate = '2016-01-01'
endDate = '2020-12-31'
tvi = 0
plateau = 7.0
msMask = 'xxxxxx0x'
indMask = '111111111111111111111111'
narMask = '1111'
dbSizes = [[msMask, indMask, str(tvi), '1']]
dbSizes.append([startDate, endDate])
dbSizes.append([])
dbSizes.append([])
#create custom db
#[0] delete old files
custTrainPath = os.path.join('..', 'data', 'ml', 'ann', 'cust', 'train')
trainFileList = os.listdir(custTrainPath)
for itrFile in trainFileList:
	itrFilePath = os.path.join(custTrainPath, itrFile)
	os.remove(itrFilePath)
custTestPath = os.path.join('..', 'data', 'ml', 'ann', 'cust', 'test')
testFileList = os.listdir(custTestPath)
for itrFile in testFileList:
	itrFilePath = os.path.join(custTestPath, itrFile)
	os.remove(itrFilePath)
#[1] get all market dates between
inRangeDates = au.getDatesBetween(startDate, endDate)
#[2] filter dates based on msMask
msDates = []
msPath = os.path.join('..', 'in', 'mstates.txt')
with open(msPath, 'r') as msFile:
	for msLine in msFile:
		lineEles = msLine.strip().split(',')
		dateItr = lineEles[0]
		msItr = lineEles[2]
		if (dateItr in inRangeDates) and (au.compareMasks(msMask, msItr)):
			msDates.append(dateItr)
print('--> All MS Dates : ', msDates)
#[3] split dates into even/odd
evenDates = []
oddDates = []
for itrDate in msDates:
	dateEles = itrDate.split('-')
	itrDay = int(dateEles[2])
	if itrDay % 2 == 0:
		evenDates.append(itrDate)	
	else:
		oddDates.append(itrDate)
#[4] div and write into train sections
bdBasePath = os.path.join('..', '..', 'DB_Intrinio', 'Clean', 'ByDate')
linesPerSection = 10048
sectionNum = 0
tfSection = []
for itrDate in evenDates:
	bdFullPath = os.path.join(bdBasePath, itrDate+'.txt')
	with open(bdFullPath, 'r') as bdFile:
		for bdLine in bdFile:
			lineEles = bdLine.strip().split('~')
			matches_nar, nline = au.normCleanLine(lineEles, tvi, plateau, indMask, narMask)
			tfLine = nline[1:]
			if matches_nar:
				#add line to sections, update if needed
				tfSection.append(tfLine)
				#print('tfLine: ', tfLine, '  |  tfSection len: ', len(tfSection))
				if len(tfSection) >= linesPerSection:
					fname = 'sec'+str(sectionNum)+'.txt'
					secPath = os.path.join('..', 'data', 'ml', 'ann', 'cust', 'train', fname)
					au.writeToFile(secPath, tfSection, ',')
					sectionNum += 1
					tfSection = []
#write remaining data in tfSection to file
if len(tfSection) > 0:
	fname = 'sec'+str(sectionNum)+'.txt'
	secPath = os.path.join('..', 'data', 'ml', 'ann', 'cust', 'train', fname)
	au.writeToFile(secPath, tfSection, ',')
	dbSizes[2].append(str(((sectionNum+1) * linesPerSection) + len(tfSection)))
	dbSizes[3].append(str(sectionNum+1))
else:
	dbSizes[2].append(str(sectionNum * linesPerSection))
	dbSizes[3].append(str(sectionNum))
sectionNum = 0
tfSection = []
#[5] same as above but for validation dataset
for itrDate in oddDates:
	bdFullPath = os.path.join(bdBasePath, itrDate+'.txt')
	with open(bdFullPath, 'r') as bdFile:
		for bdLine in bdFile:
			lineEles = bdLine.strip().split('~')
			matches_nar, nline = au.normCleanLine(lineEles, tvi, plateau, indMask, narMask)
			tfLine = nline[1:]
			if matches_nar:
				#add line to sections, update if needed
				tfSection.append(tfLine)
				#print('tfLine: ', tfLine, '  |  tfSection len: ', len(tfSection))
				if len(tfSection) >= linesPerSection:
					fname = 'sec'+str(sectionNum)+'.txt'
					secPath = os.path.join('..', 'data', 'ml', 'ann', 'cust', 'train', fname)
					au.writeToFile(secPath, tfSection, ',')
					sectionNum += 1
					tfSection = []
#write remaining data in tfSection to file
if len(tfSection) > 0:
	fname = 'sec'+str(sectionNum)+'.txt'
	secPath = os.path.join('..', 'data', 'ml', 'ann', 'cust', 'test', fname)
	au.writeToFile(secPath, tfSection, ',')
	dbSizes[2].append(str(((sectionNum+1) * linesPerSection) + len(tfSection)))
	dbSizes[3].append(str(sectionNum+1))
else:
	dbSizes[2].append(str(sectionNum * linesPerSection))
	dbSizes[3].append(str(sectionNum))
sectionNum = 0
tfSection = []
#[6] write db_sizes data to file
dbSizesPath = os.path.join('..', 'data', 'ml', 'ann', 'cust', 'db_sizes.txt')
au.writeToFile(dbSizesPath, dbSizes, ',')
#[7] save train / validation files
trainFileList = os.listdir(custTrainPath)
testFileList = os.listdir(custTestPath)

						
#======= Neural Network Stuff =======

#set hyperparams
inputSize = 24
hiddenSize = 48
outputSize = 1
learnRate = 0.01
numOfEpochs = 3
batchSize = 64

#create instance of neural network
mynn = StagNN.H1NN(inputSize, hiddenSize, outputSize)

#create loss funct and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(mynn.parameters(), lr=learnRate)

#train the neural network
avgTrainErr = 0.0
avgValidErr = 0.0
trainCount = 0
validCount = 0
for epoch in range(numOfEpochs):
	for i in range(len(trainFileList)):
		#read in data from sec file and translate to tensor
		fsTensor, targetTensor = fileToTensors(os.path.join(custTrainPath, trainFileList[i]), batchSize)
		for j in range(len(fsTensor)):
			#forward pass
			output = mynn(fsTensor[j])
			loss = criterion(output, targetTensor[j])
			#backward pass and optimization
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			avgTrainErr += loss.item()
			trainCount += 1
			if (trainCount % 200) == 0:
				print(f'--> Train Loss At {trainCount} : {loss.item()}')

		#print(f'Epoch: {epoch+1}, Section: {i}, Batch: {trainCount}') 

		#run through a test section file
		if i < len(testFileList):
			testFilePath = os.path.join(custTestPath, testFileList[i])
			if os.path.exists(testFilePath):
				fsTensor, targetTensor = fileToTensors(testFilePath, 10000)
				#forward pass
				with torch.no_grad():
					output = mynn(fsTensor[0])
					loss = criterion(output, targetTensor[0])
				avgValidErr += loss.item()
				validCount += 1	
				#print(f'Epoch: {epoch+1}, Section: {i}')
				#print(f'--> Valid Loss: {loss.item()}')
avgTrainErr = avgTrainErr / trainCount
avgValidErr = avgValidErr / validCount
nodeIdx = 44
itrBias = mynn.hidden1.bias.data[nodeIdx]
itrWeights = mynn.hidden1.weight.data[nodeIdx]
print(f'--> HL Node {nodeIdx} Bias : ', itrBias)
print(f'--> HL Node {nodeIdx} Weights : ', itrWeights)
print(f'--> Avg Train Loss: {avgTrainErr:.7f}')
print(f'--> Avg Valid Loss: {avgValidErr:.7f}')
saveStr = input('Save this ANN Model? (y/n) : ')
if saveStr.lower() == 'y':
	torch.save(mynn.state_dict(), 'mynn.pt')
	hparams = []
	hparams.append(['sdate', startDate])
	hparams.append(['edate', endDate])
	hparams.append(['tvi', str(tvi)])
	hparams.append(['plateau', str(plateau)])
	hparams.append(['ms_mask', msMask])
	hparams.append(['ind_mask', indMask])
	hparams.append(['nar_mask', narMask])
	au.writeToFile('hparams.txt', hparams, ':')

#torch.save(optimizer.state_dict(), 'myopt.pt')
			

#test the model
#with torch.no_grad():
#	predictions = model(validTensor)
#	print(f'Predictions: {predictions}')

