import os
import sys
import math
from datetime import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
import StagNN as stag
import AhrUtil as au
import TrainedKey as tkey
from FCI import FCI

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

#check all data in an ANN for nan values
def checkNansInNN(mynn):
	#[0] HL weights, [1] HL biases, [2] OL weights, [3] OL biases
	has_nans = [False, False, False, False]
	has_nans[0] = torch.isnan(mynn.hidden1.weight).any().item()
	has_nans[1] = torch.isnan(mynn.hidden1.bias).any().item()
	has_nans[2] = torch.isnan(mynn.output.weight).any().item()
	has_nans[3] = torch.isnan(mynn.output.bias).any().item()
	return has_nans


#======= Data Management =======

#define vars for custom db
startDate = '2018-01-01'
endDate = '2023-06-01'
activationFunctCode = 'SIGM'
lossFunct = 0
spd = 10
tvi = 4
plateau = 15.0
msMask = 'xxxxxxx0'
indMask = '111111111111111111111111'
narMask = '1111x'
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
custTestPath = os.path.join('..', 'data', 'ml', 'ann', 'cust', 'valid')
testFileList = os.listdir(custTestPath)
for itrFile in testFileList:
	itrFilePath = os.path.join(custTestPath, itrFile)
	os.remove(itrFilePath)
#[1] get all market dates between
inRangeDates = au.getDatesBetween(startDate, endDate)
#[2] filter dates based on msMask
msDates = []
msPath = os.path.join('.', '..', 'in', 'mstates.txt')
fciMS = FCI(False, msPath)
with open(msPath, 'r') as msFile:
	for msLine in msFile:
		lineEles = msLine.strip().split(',')
		dateItr = lineEles[fciMS.getIdx('date')]
		msItr = lineEles[fciMS.getIdx('ms_mask')]
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
tvVals = []
for itrDate in evenDates:
	bdFullPath = os.path.join(bdBasePath, itrDate+'.txt')
	with open(bdFullPath, 'r') as bdFile:
		for bdLine in bdFile:
			lineEles = bdLine.strip().split('~')
			matches_nar, nline, tvActStr = au.normCleanLine(lineEles, tvi, plateau, indMask, narMask)
			tfLine = nline[1:]
			if matches_nar:
				#track TV value
				tvVals.append(float(nline[-1]))
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
			matches_nar, nline, tvActStr = au.normCleanLine(lineEles, tvi, plateau, indMask, narMask)
			tfLine = nline[1:]
			if matches_nar:
				#add line to sections, update if needed
				tfSection.append(tfLine)
				#print('tfLine: ', tfLine, '  |  tfSection len: ', len(tfSection))
				if len(tfSection) >= linesPerSection:
					fname = 'sec'+str(sectionNum)+'.txt'
					secPath = os.path.join('..', 'data', 'ml', 'ann', 'cust', 'valid', fname)
					au.writeToFile(secPath, tfSection, ',')
					sectionNum += 1
					tfSection = []
#write remaining data in tfSection to file
if len(tfSection) > 0:
	fname = 'sec'+str(sectionNum)+'.txt'
	secPath = os.path.join('..', 'data', 'ml', 'ann', 'cust', 'valid', fname)
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
		#print('===== AhrLoss =====')
		#print('-> predVals mean = ', torch.mean(predVals))
		#print('-> predVals stddev = ', torch.std(predVals))
		#print('-> predVals = ', predVals)
		#print('-> actVals mean = ', torch.mean(actVals))
		#print('-> actVals stddev = ', torch.std(actVals))
		#print('-> actVals = ', actVals)
		#slopes and intercepts
		m1 = 1.0
		b1 = 0.0
		m2 = -1.0 / m1
		b2 = torch.sub(predVals, (m2 * actVals))
		#print('-> b2 = ', b2)
		#calc intersection of the 2 lines
		intersectionX = (b2 - b1) / (m1 - m2)
		intersectionY = (m1 * intersectionX) + b1
		#print('-> intersectionX = ', intersectionX, '\n-> intersectionY = ', intersectionY)
		#pythagorean theorem
		distX = torch.sub(intersectionX, actVals) ** 2
		distY = torch.sub(intersectionY, predVals) ** 2
		#print('-> distX = ', distX, '\n-> distY = ', distY)
		distance = torch.add(distX, distY)
		#print('-> distance (1) = ', distance)
		distance = torch.sqrt(distance)
		#print('-> distance (2) = ', distance)
		#mimic math function : y = | -(x^2) + 0.1767766953 |
		xic = 0.420488
		yic = 0.1767766953
		newPV = predVals * (xic * 2.0)
		#print('-> newPV (1) = ', newPV)
		newPV = newPV - xic
		#print('-> newPV (2) = ', newPV)
		extraErr = torch.abs((-1.0 * (newPV ** 2)) + yic)
		#print('-> extraErr = ', extraErr)
		#calc loss
		loss = torch.add(distance, extraErr)
		#print('-> loss = ', loss)
		#lossScalar = torch.sum(loss) / loss.numel()
		sumOfLoss = torch.sum(loss)
		numOfEles = loss.numel()
		lossScalar = sumOfLoss / numOfEles
		#print(f'-> lossScalar = {sumOfLoss:.5f} / {numOfEles} = {lossScalar:.5f}')
		#print('====================')
		#print('lossScalar : ', lossScalar)
		if (0.0 <= lossScalar <= 15.0):
			return lossScalar
		else:
			sys.exit()


#get user selection for ANN type
#maybe always do both?
promptIn = """--> Select ANN type to train ...
    1) Regression
    2) Classification
Enter: """
#pick = int(input(promptIn))
#is_regressor = True
#if pick == 2:
#	is_regressor = False

#calc nodes in inputlayer & hiddenlayer
inputSize = indMask.count('1')
regHiddenSizes = [(inputSize * 2)]
clsHiddenSizes = [55]
regOutputSize = 1
clsOutputSize = 3
#other hyperparams
learnRate = 0.007
numOfEpochs = 7
trainBatchSize = 64
validBatchSize = linesPerSection
regErrLog = []
clsMeta = torch.zeros(clsOutputSize,3)
clsErrLog = []
#calc classification NN threshold lvls (according to clsOutputSize)
tvVals = sorted(tvVals)
clsStepSize = float(len(tvVals) / clsOutputSize)
clsIncrement = 0.0
clsThresholds = []
for i in range(clsOutputSize-1):
	clsIncrement += clsStepSize
	clsIdx = round(clsIncrement-1.0)
	clsThresholds.append(tvVals[clsIdx])

#create instance of neural network
regNN = stag.Regressor1(inputSize, regHiddenSizes, regOutputSize)
clsNN = stag.ClassifierX(inputSize, clsHiddenSizes, clsOutputSize)
#au.inDepthDir('mynn', mynn)

#create loss funct and optimizer
criterionMSE = nn.MSELoss()
#regCriterion = AhrLoss()
regCriterion = nn.MSELoss()
#au.inDepthDir('criterion Ahr', regCriterion)
#au.inDepthDir('criterion MSE', criterionMSE)
clsCriterion = nn.CrossEntropyLoss()

regOptimizer = optim.SGD(regNN.parameters(), lr=learnRate)
clsOptimizer = optim.SGD(clsNN.parameters(), lr=learnRate)

#clear grad desc debug file
open('debug_gd.txt', 'w').close()
debugLineCount = 0
#train the neural network
avgTrainErr = 0.0
avgValidErr = 0.0
trainBatchCount = 0
validBatchCount = 0
trainLineCount = 0
validLineCount = 0
for epoch in range(numOfEpochs):
	fstCount = 0	#feature space tensor count (for gd debug)
	for i in range(len(trainFileList)):
		print(f'=========== Train File {i} (epoch:{epoch}) ==========')
		#read in data from sec file and translate to tensor
		fspaceTensor, regTargetTensor = fileToTensors(os.path.join(custTrainPath, trainFileList[i]), trainBatchSize)
		clsTargetTensor = au.binTargetTensor(regTargetTensor, clsThresholds)
		print(f'fspaceTensor shape : {fspaceTensor.shape}')
		print(f'fspaceTensor has nan : {torch.isnan(fspaceTensor).any().item()}')
		print(f'regTargetTensor shape : {regTargetTensor.shape}')
		print(f'regTargetTensor has nan : {torch.isnan(regTargetTensor).any().item()}')
		print(f'clsTargetTensor shape : {clsTargetTensor.shape}')
		print(f'clsTargetTensor has nan : {torch.isnan(clsTargetTensor).any().item()}')
		for j in range(len(fspaceTensor)):
			#print(f'--> fspaceTensor[{j}] shape : {fspaceTensor[j].shape}')
			regTT = regTargetTensor[j]
			clsTT = clsTargetTensor[j].squeeze().to(torch.long)

			#========= DEBUG ===========
			debugLineCount += 1
			#print(f'--> Before FF : {checkNansInNN(regNN)}')
			hasNans = checkNansInNN(regNN)
			debugLineStr = ''
			debugLineStr += str(debugLineCount)+','
			debugLineStr += str(epoch)+','
			debugLineStr += str(fstCount)+','
			debugLineStr += '0,'
			debugLineStr += str(hasNans[0])+','
			debugLineStr += str(hasNans[1])+','
			debugLineStr += str(hasNans[2])+','
			debugLineStr += str(hasNans[3])+','
			for name, param in regNN.named_parameters():
				if param.grad is not None:
					debugLineStr += f'{param.grad.norm().item():.5f},'
				else:
					debugLineStr += '0.0,'
			debugLineStr = debugLineStr[:-1]
			debugLineStr += '\n'
			with open('debug_gd.txt', 'a') as debugFile:
				debugFile.write(debugLineStr)	
			#===========================

			#forward pass
			regOut = regNN(fspaceTensor[j], activationFunctCode)
			if (torch.isnan(regOut).any().item()):
				print('!!! regOut has nan values !!!')

			#========= DEBUG ===========
			debugLineCount += 1
			#print(f'--> After FF : {checkNansInNN(regNN)}')
			hasNans = checkNansInNN(regNN)
			debugLineStr = ''
			debugLineStr += str(debugLineCount)+','
			debugLineStr += str(epoch)+','
			debugLineStr += str(fstCount)+','
			debugLineStr += '1,'
			debugLineStr += str(hasNans[0])+','
			debugLineStr += str(hasNans[1])+','
			debugLineStr += str(hasNans[2])+','
			debugLineStr += str(hasNans[3])+','
			for name, param in regNN.named_parameters():
				if param.grad is not None:
					debugLineStr += f'{param.grad.norm().item():.5f},'
				else:
					debugLineStr += '0.0,'
			debugLineStr = debugLineStr[:-1]
			debugLineStr += '\n'
			with open('debug_gd.txt', 'a') as debugFile:
				debugFile.write(debugLineStr)	
			#===========================		

			clsOut = clsNN(fspaceTensor[j])
			regLoss = regCriterion(regOut, regTT)
			clsLoss = clsCriterion(clsOut, clsTT)

			#========= DEBUG ===========
			debugLineCount += 1
			#print(f'--> After criterion : {checkNansInNN(regNN)}')
			hasNans = checkNansInNN(regNN)
			debugLineStr = ''
			debugLineStr += str(debugLineCount)+','
			debugLineStr += str(epoch)+','
			debugLineStr += str(fstCount)+','
			debugLineStr += '2,'
			debugLineStr += str(hasNans[0])+','
			debugLineStr += str(hasNans[1])+','
			debugLineStr += str(hasNans[2])+','
			debugLineStr += str(hasNans[3])+','
			for name, param in regNN.named_parameters():
				if param.grad is not None:
					debugLineStr += f'{param.grad.norm().item():.5f},'
				else:
					debugLineStr += '0.0,'
			debugLineStr = debugLineStr[:-1]
			debugLineStr += '\n'
			with open('debug_gd.txt', 'a') as debugFile:
				debugFile.write(debugLineStr)	
			#===========================

			#print(f'--> Epoch {epoch}, Loss : {regLoss.item()}')
			#backward pass and optimization
			regOptimizer.zero_grad()
			clsOptimizer.zero_grad()

			#========= DEBUG ===========
			'''
			debugLineCount += 1
			print(f'--> After zero grad : {checkNansInNN(regNN)}')
			hasNans = checkNansInNN(regNN)
			debugLineStr = ''
			debugLineStr += str(debugLineCount)+','
			debugLineStr += str(epoch)+','
			debugLineStr += str(fstCount)+','
			debugLineStr += '3,'
			debugLineStr += str(hasNans[0])+','
			debugLineStr += str(hasNans[1])+','
			debugLineStr += str(hasNans[2])+','
			debugLineStr += str(hasNans[3])+','
			for name, param in regNN.named_parameters():
				if param.grad is not None:
					debugLineStr += f'{param.grad.norm().item():.5f},'
			debugLineStr = debugLineStr[:-1]
			debugLineStr += '\n'
			with open('debug_gd.txt', 'a') as debugFile:
				debugFile.write(debugLineStr)	
			'''
			#===========================

			regLoss.backward()
			#for name, param in regNN.named_parameters():
			#	if param.grad is not None:
			#		print(f'    Parameter: {name}, Grad Norm: {param.grad.norm().item()} | {param.data.tolist()}')
			clsLoss.backward()

			gd_is_nan = False
			for name, param in regNN.named_parameters():
				if param.grad.norm() != param.grad.norm():
					gd_is_nan = True
			if not gd_is_nan:
				#========= DEBUG ===========
				debugLineCount += 1
				#print(f'--> After backward : {checkNansInNN(regNN)}')
				hasNans = checkNansInNN(regNN)
				debugLineStr = ''
				debugLineStr += str(debugLineCount)+','
				debugLineStr += str(epoch)+','
				debugLineStr += str(fstCount)+','
				debugLineStr += '4,'
				debugLineStr += str(hasNans[0])+','
				debugLineStr += str(hasNans[1])+','
				debugLineStr += str(hasNans[2])+','
				debugLineStr += str(hasNans[3])+','
				for name, param in regNN.named_parameters():
					if param.grad is not None:
						debugLineStr += f'{param.grad.norm().item():.5f},'
					else:
						debugLineStr += '0.0,'
				debugLineStr = debugLineStr[:-1]
				debugLineStr += '\n'
				with open('debug_gd.txt', 'a') as debugFile:
					debugFile.write(debugLineStr)	
				#===========================

				#print gradient val
				#for name, param in regNN.named_parameters():
				#	if param.grad is not None:
				#		print(f'    Parameter: {name}, Grad Norm: {param.grad.norm().item()} | {param.data.tolist()}')
						#print(f'    Parameter: {name}, Grad Norm: {param.grad.norm().item()}, Param Val: {param.data}')
						#print(f'    Parameter: {name}, Grad Norm: {param.grad}, Param Val: {param.data.tolist()}')
				#torch.nn.utils.clip_grad_norm_(regNN.parameters(), max_norm=0.5)

				regOptimizer.step() 	#nan vals start here!
				#for name, param in regNN.named_parameters():
				#	if param.grad is not None:
				#		print(f'    Parameter: {name}, Grad Norm: {param.grad.norm().item()} | {param.data.tolist()}')
				clsOptimizer.step()

				#========= DEBUG ===========
				debugLineCount += 1
				#print(f'--> After opt step : {checkNansInNN(regNN)}')
				hasNans = checkNansInNN(regNN)
				debugLineStr = ''
				debugLineStr += str(debugLineCount)+','
				debugLineStr += str(epoch)+','
				debugLineStr += str(fstCount)+','
				debugLineStr += '5,'
				debugLineStr += str(hasNans[0])+','
				debugLineStr += str(hasNans[1])+','
				debugLineStr += str(hasNans[2])+','
				debugLineStr += str(hasNans[3])+','
				for name, param in regNN.named_parameters():
					if param.grad is not None:
						debugLineStr += f'{param.grad.norm().item():.5f},'
					else:
						debugLineStr += '0.0,'
				debugLineStr = debugLineStr[:-1]
				debugLineStr += '\n'
				with open('debug_gd.txt', 'a') as debugFile:
					debugFile.write(debugLineStr)	
				#===========================
	
				#update error for regression model
				avgTrainErr += regLoss.item()
				trainBatchCount += 1
				trainLineCount += trainBatchSize
				if ((trainBatchCount - 1) % 250) == 0:
					regErrLogLine = []
					regErrLogLine.append('train')
					regErrLogLine.append(str(trainBatchCount))
					regErrLogLine.append(str(trainLineCount))
					regErrLogLine.append(f"{regLoss.item():.7f}")
					regErrLog.append(regErrLogLine)
					print(f'--> Reg Train Loss At {trainLineCount} : {regLoss.item()}')
				#update error log for classification model
				for k in range(len(clsTT)):
					actualBin = clsTT[k]
					predBin, is_right_prediction = au.isRightClassificationPrediction(clsOut[k], actualBin)
					clsMeta[actualBin][0] += 1
					clsMeta[predBin][1] += 1
					if is_right_prediction:
						clsMeta[actualBin][2] += 1
				if ((trainBatchCount - 1) % 250) == 0:
					clsErrLogLine = []
					clsErrLogLine.append('train')
					clsErrLogLine.append(str(trainLineCount))
					for mline in clsMeta:
						for ele in mline:
							clsErrLogLine.append(str(ele.item()))
					clsErrLog.append(clsErrLogLine)
					#print(f'--> Cls Train Loss At {trainLineCount} : {clsLoss.item()}')
					#print(f'--> clsMeta : {clsMeta.tolist()}')

				#========= DEBUG ===========
				debugLineCount += 1
				#print(f'--> Last line in train : {checkNansInNN(regNN)}')
				hasNans = checkNansInNN(regNN)
				debugLineStr = ''
				debugLineStr += str(debugLineCount)+','
				debugLineStr += str(epoch)+','
				debugLineStr += str(fstCount)+','
				debugLineStr += '6,'
				debugLineStr += str(hasNans[0])+','
				debugLineStr += str(hasNans[1])+','
				debugLineStr += str(hasNans[2])+','
				debugLineStr += str(hasNans[3])+','
				for name, param in regNN.named_parameters():
					if param.grad is not None:
						debugLineStr += f'{param.grad.norm().item():.5f},'
					else:
						debugLineStr += '0.0,'
				debugLineStr = debugLineStr[:-1]
				debugLineStr += '\n'
				with open('debug_gd.txt', 'a') as debugFile:
					debugFile.write(debugLineStr)	
				fstCount += 1
				#===========================
			else:
				fstCount += 1
			
		#run through a test section file
		if i < len(testFileList):
			testFilePath = os.path.join(custTestPath, testFileList[i])
			if os.path.exists(testFilePath):
				fspaceTensor, regTargetTensor = fileToTensors(testFilePath, validBatchSize)
				clsTargetTensor = au.binTargetTensor(regTargetTensor, clsThresholds)
				clsTT = clsTargetTensor[0].squeeze().to(torch.long)
				#forward pass
				with torch.no_grad():
					regOut = regNN(fspaceTensor[0], activationFunctCode)
					clsOut = clsNN(fspaceTensor[0])
					regLoss = regCriterion(regOut, regTargetTensor[0])
					clsLoss = clsCriterion(clsOut, clsTT)
					#loss = custLoss(output, targetTensor[0])
				avgValidErr += regLoss.item()
				validBatchCount += 1
				validLineCount += validBatchSize
				regErrLogLine = []
				regErrLogLine.append('valid')
				regErrLogLine.append(str(validBatchCount))
				regErrLogLine.append(str(validLineCount))
				regErrLogLine.append(f"{regLoss.item():.7f}")
				regErrLog.append(regErrLogLine)
				#print(f'--> Reg Valid Loss At {validLineCount} : {regLoss.item()}')
				#print(f'--> Cls Valid Loss At {validLineCount} : {clsLoss.item()}')
avgTrainErr = avgTrainErr / trainBatchCount
avgValidErr = avgValidErr / validBatchCount
nodeIdx = 7
itrBias = regNN.hidden1.bias.data[nodeIdx]
itrWeights = regNN.hidden1.weight.data[nodeIdx]
print(f'--> HL Node {nodeIdx} Bias : {itrBias}')
print(f'--> HL Node {nodeIdx} Weights : {itrWeights}')
print(f'--> Avg Train Loss: {avgTrainErr:.7f}')
print(f'--> Avg Valid Loss: {avgValidErr:.7f}')
saveStr = input('Save this ANN Model? (y/n) : ')
if saveStr.lower() == 'y':
	#get SK num
	ksPath = os.path.join('.', '..', 'out', 'sk', 'log', 'ann', 'keys_struct.txt')
	fciKS = FCI(True, ksPath)
	ksFile = []
	keysInKS = []
	with open(ksPath, 'r') as itrFile:
		header = itrFile.readline().strip().split(',')
		for itrLine in itrFile:
			data = itrLine.strip().split(',')
			ksFile.append(data)
			keysInKS.append(int(data[fciKS.getIdx('sk_num')]))
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
	sline.append(str(newSK))							#[0] sk_num
	sline.append('python')								#[1] language
	sline.append('IT')									#[2] db_used
	sline.append(dt.today().strftime('%Y-%m-%d'))		#[3] date_ran
	sline.append(startDate)								#[4] start_date
	sline.append(endDate)								#[5] end_date
	sline.append(str(numOfEpochs))						#[6] epochs
	sline.append(str(trainBatchSize))					#[7] batch_size
	hlDims = ""
	for i in range(len(regHiddenSizes)):
		hlDims += str(regHiddenSizes[i])
		if (i != (len(regHiddenSizes)-1)):
			hlDims += "~"
	sline.append(hlDims)								#[8] hidden_layer_dims
	sline.append(activationFunctCode)					#[9] activation_funct
	sline.append(str(lossFunct))						#[10] loss_funct
	sline.append('CR')									#[11] output_type
	sline.append('0')									#[12] call
	sline.append(f"{learnRate:.5f}")					#[13] learn_rate
	sline.append(f"{plateau:.2f}")						#[14] plateau
	sline.append(str(spd))								#[15] spd
	sline.append(str(tvi))								#[16] tvi
	sline.append(msMask)								#[17] ms_mask
	sline.append(indMask)								#[18] ind_mask
	sline.append(narMask)								#[19] nar_mask
	sline.append(f"{avgValidErr:.11f}")					#[20] avg_error
	lline = sline.copy()
	lline[0] = str(newSK+1)
	lline[fciKS.getIdx('call')] = '1'
	ksFile.append(sline)
	ksFile.append(lline)
	au.writeToFile(ksPath, ksFile, ',')
	print('--> ', ksPath, ' WRITTEN')
	#write data to keys_perf file
	sline = []
	sline.append(str(newSK))			#[0] sk_num
	sline.append('0')					#[1] call
	sline.append(str(spd))				#[2] spd
	sline.append(str(tvi))				#[3] tvi
	sline.append(msMask)				#[4] ms_mask
	sline.append(narMask)				#[5] nar_mask
	sline.append('ph')					#[6] bim
	sline.append('ph')					#[7] som
	sline.append('ph')					#[8] bso_train_apapt
	sline.append('ph')					#[9] bso_valid_apapt
	sline.append('ph')					#[10] bso_train_posp
	sline.append('ph')					#[11] bso_valid_posp
	sline.append('ph')					#[12] plat_train_apapt
	sline.append('ph')					#[13] plat_valid_apapt
	sline.append('ph')					#[14] true_train_apapt
	sline.append('ph')					#[15] true_valid_apapt
	sline.append('ph')					#[16] true_train_posp
	sline.append('ph')					#[17] true_valid_posp
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
	torch.save(regNN.state_dict(), structPath1)
	torch.save(regNN.state_dict(), structPath2)
	print('--> ', structPath1, ' WRITTEN')
	print('--> ', structPath2, ' WRITTEN')
	#write to error log file
	au.writeToFile(errorPath1, regErrLog, ',')
	au.writeToFile(errorPath2, regErrLog, ',')
	print('--> ', errorPath1, ' WRITTEN')
	print('--> ', errorPath2, ' WRITTEN')
	#write to basis file
	skShort = tkey.SingleKey(str(newSK))
	skShort.createBasisFile()
	print(f'--> ./../out/sk/baseis/ann/ANN_{str(newSK)}.txt ... WRITTEN')
	skLong = tkey.SingleKey(str(newSK+1))	
	skLong.createBasisFile()
	print(f'--> ./../out/sk/baseis/ann/ANN_{str(newSK+1)}.txt ... WRITTEN')
	#calc perf and replace PH w/ real vals in keys_perf
	skShort.calcKeysPerf()
	skLong.calcKeysPerf()
	print(f'--> {kpPath} ... UPDATED')
else:
	errorPath3 = os.path.join('..', 'out', 'cls_err_log.txt')
	au.writeToFile(errorPath3, clsErrLog, ',')
	print('--> ', errorPath3, ' WRITTEN')
print('--> Trainer.py ... COMPLETE')



#torch.save(optimizer.state_dict(), 'myopt.pt')
			

#test the model
#with torch.no_grad():
#	predictions = model(validTensor)
#	print(f'Predictions: {predictions}')

