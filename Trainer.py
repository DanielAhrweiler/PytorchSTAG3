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
	fspace = []
	target = []
	fspaceBatch = []
	targetBatch = []
	with open(fpath, 'r') as itrFile:
		for itrLine in itrFile:
			data = itrLine.strip().split(',')
			fspaceBatch.append(list(map(float, data[:(len(data)-1)])))
			targetBatch.append([float(data[len(data)-1])])
			if (len(targetBatch) >= batchSize):
				fspace.append(fspaceBatch)
				target.append(targetBatch)
				fspaceBatch = []
				targetBatch = []
	if (len(fspace) == 0):
		fspace.append(fspaceBatch)
		target.append(targetBatch)
	fspaceTensor = torch.tensor(fspace)
	targetTensor = torch.tensor(target)
	return fspaceTensor, targetTensor


#======= Data Management =======

#define vars for custom db
startDate = '2016-01-01'
endDate = '2020-12-31'
spd = 10
tvi = 6
plateau = 15.0
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
#print('--> All MS Dates : ', msDates)
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
					secPath = os.path.join('..', 'data', 'ml', 'ann', 'cust', 'test', fname)
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

#custom loss in class form
class AhrLoss(nn.Module):
	def __init__(self):
		super(AhrLoss, self).__init__()

	def forward(self, predVals, actVals):
		#slopes and intercepts
		m1 = 1.0
		b1 = 0.0
		m2 = -1.0 / m1
		b2 = torch.sub(predVals, (m2 * actVals))
		#calc intersection of the 2 lines
		intersectionX = (b2 - b1) / (m1 - m2)
		intersectionY = (m1 * intersectionX) + b1
		#pythagorean theorem
		distX = torch.sub(intersectionX, actVals) ** 2
		distY = torch.sub(intersectionY, predVals) ** 2
		distance = torch.add(distX, distY)
		distance = torch.sqrt(distance)
		#mimic math function : y = | -(x^2) + 0.1767766953 |
		xic = 0.420488
		yic = 0.1767766953
		newPV = predVals * (xic * 2.0)
		newPV = newPV - xic
		extraErr = torch.abs((-1.0 * (newPV ** 2)) + yic)
		#calc loss
		loss = torch.add(distance, extraErr)
		lossScalar = torch.sum(loss) / loss.numel()
		#print('lossScalar : ', lossScalar)
		return lossScalar



#calc nodes in inputlayer & hiddenlayer
inputSize = indMask.count('1')
hiddenSize = inputSize * 2
outputSize = 1
#other hyperparams
learnRate = 0.001
numOfEpochs = 3
trainBatchSize = 64
validBatchSize = linesPerSection
errLog = []

#create instance of neural network
mynn = StagNN.H1NN(inputSize, hiddenSize, outputSize)
#au.inDepthDir('mynn', mynn)

#create loss funct and optimizer
#criterionMSE = nn.MSELoss()
#au.inDepthDir('criterion MSE', criterionMSE)
criterion = AhrLoss()
#au.inDepthDir('criterion Ahr', criterion)

optimizer = optim.SGD(mynn.parameters(), lr=learnRate)

#train the neural network
avgTrainErr = 0.0
avgValidErr = 0.0
trainBatchCount = 0
validBatchCount = 0
trainLineCount = 0
validLineCount = 0
for epoch in range(numOfEpochs):
	for i in range(len(trainFileList)):
		#read in data from sec file and translate to tensor
		fspaceTensor, targetTensor = fileToTensors(os.path.join(custTrainPath, trainFileList[i]), trainBatchSize)
		#fspaceTensor.requires_grad_(True)
		#print('fspaceTensor requires_grad = ', fspaceTensor.requires_grad)
		#print('fspaceTensor grad_fn = ', fspaceTensor.grad_fn)
		for j in range(len(fspaceTensor)):
			#forward pass
			output = mynn(fspaceTensor[j])
			print('An output : ', output[-1, -1])
			#print('output requires_grad = ', output.requires_grad)
			#print('output grad_fn = ', output.grad_fn)
			#print('targetTensor requires_grad = ', targetTensor.requires_grad)
			#print('targetTensor grad_fn = ', targetTensor.grad_fn)
			loss = criterion(output, targetTensor[j])
			#loss = custLoss(output, targetTensor[j])
			#backward pass and optimization
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			avgTrainErr += loss.item()
			trainBatchCount += 1
			trainLineCount += trainBatchSize
			if ((trainBatchCount - 1) % 200) == 0:
				errLogLine = []
				errLogLine.append('train')
				errLogLine.append(str(trainBatchCount))
				errLogLine.append(str(trainLineCount))
				errLogLine.append(f"{loss.item():.7f}")
				errLog.append(errLogLine)
				print(f'--> Train Loss At {trainLineCount} : {loss.item()}')
		#run through a test section file
		if i < len(testFileList):
			testFilePath = os.path.join(custTestPath, testFileList[i])
			if os.path.exists(testFilePath):
				fspaceTensor, targetTensor = fileToTensors(testFilePath, validBatchSize)
				#forward pass
				with torch.no_grad():
					output = mynn(fspaceTensor[0])
					loss = criterion(output, targetTensor[0])
					#loss = custLoss(output, targetTensor[0])
				avgValidErr += loss.item()
				validBatchCount += 1
				validLineCount += validBatchSize	
				errLogLine = []
				errLogLine.append('valid')
				errLogLine.append(str(validBatchCount))
				errLogLine.append(str(validLineCount))
				errLogLine.append(f"{loss.item():.7f}")
				errLog.append(errLogLine)
				print(f'--> Valid Loss At {validLineCount} : {loss.item()}')
				#print(f'Epoch: {epoch+1}, Section: {i}')
				#print(f'--> Valid Loss: {loss.item()}')
avgTrainErr = avgTrainErr / trainBatchCount
avgValidErr = avgValidErr / validBatchCount
nodeIdx = 7
itrBias = mynn.hidden1.bias.data[nodeIdx]
itrWeights = mynn.hidden1.weight.data[nodeIdx]
print(f'--> HL Node {nodeIdx} Bias : ', itrBias)
print(f'--> HL Node {nodeIdx} Weights : ', itrWeights)
print(f'--> Avg Train Loss: {avgTrainErr:.7f}')
print(f'--> Avg Valid Loss: {avgValidErr:.7f}')
saveStr = input('Save this ANN Model? (y/n) : ')
if saveStr.lower() == 'y':
	#get SK num
	ksPath = os.path.join('..', 'out', 'sk', 'log', 'ann', 'keys_struct.txt')
	ksFile = []
	keysInKS = []
	with open(ksPath, 'r') as itrFile:
		header = itrFile.readline().strip().split(',')
		for itrLine in itrFile:
			data = itrLine.strip().split(',')
			ksFile.append(data)
			keysInKS.append(int(data[0]))
	ksFile.insert(0, header)
	newSK = max(keysInKS)
	newSK = newSK + 1
	#other file paths
	kpPath = os.path.join('..', 'out', 'sk', 'log', 'ann', 'keys_perf.txt')
	structPath1 = os.path.join('..', 'out', 'sk', 'log', 'ann', 'structure', 'struct_'+str(newSK)+'.pt')	
	structPath2 = os.path.join('..', 'out', 'sk', 'log', 'ann', 'structure', 'struct_'+str(newSK+1)+'.pt')	
	errorPath1 = os.path.join('..', 'out', 'sk', 'log', 'ann', 'error', 'err_'+str(newSK)+'.txt')
	errorPath2 = os.path.join('..', 'out', 'sk', 'log', 'ann', 'error', 'err_'+str(newSK+1)+'.txt')


	#write data to keys_struct file
	sline = []
	sline.append(str(newSK))
	sline.append('python')
	sline.append('IT')
	sline.append('CR')
	sline.append(dt.today().strftime('%Y-%m-%d'))
	sline.append(startDate)
	sline.append(endDate)
	sline.append('0')
	sline.append(f"{learnRate:.5f}")
	sline.append(f"{plateau:.2f}")
	sline.append(str(spd))
	sline.append(str(tvi))
	sline.append(msMask)
	sline.append(indMask)
	sline.append(narMask)
	sline.append(f"{avgValidErr:.11f}")
	lline = sline.copy()
	lline[0] = str(newSK+1)
	lline[7] = '1'
	ksFile.append(sline)
	ksFile.append(lline)
	au.writeToFile(ksPath, ksFile, ',')
	print('--> ', ksPath, ' WRITTEN')
	#write data to keys_perf file
	sline = []
	sline.append(str(newSK))
	sline.append('0')
	sline.append(str(spd))
	sline.append(str(tvi))
	sline.append(msMask)
	sline.append(narMask)
	sline.append('ph')
	sline.append('ph')
	sline.append('ph')
	sline.append('ph')
	sline.append('ph')
	sline.append('ph')
	sline.append('ph')
	sline.append('ph')
	sline.append('ph')
	sline.append('ph')
	sline.append('ph')
	sline.append('ph')
	lline = sline.copy()
	lline[0] = str(newSK+1)
	lline[1] = '1'
	slineStr = ','.join(sline)
	llineStr = ','.join(lline)
	appendStr = slineStr + '\n' + llineStr + '\n'
	with open(kpPath, 'a') as kpFile:
		kpFile.write(appendStr)
	print('--> ', kpPath, ' WRITTEN')
	#write data to struct file
	torch.save(mynn.state_dict(), structPath1)
	torch.save(mynn.state_dict(), structPath2)
	print('--> ', structPath1, ' WRITTEN')
	print('--> ', structPath2, ' WRITTEN')
	#write to error log file
	au.writeToFile(errorPath1, errLog, ',')
	au.writeToFile(errorPath2, errLog, ',')
	print('--> ', errorPath1, ' WRITTEN')
	print('--> ', errorPath2, ' WRITTEN')



#torch.save(optimizer.state_dict(), 'myopt.pt')
			

#test the model
#with torch.no_grad():
#	predictions = model(validTensor)
#	print(f'Predictions: {predictions}')

