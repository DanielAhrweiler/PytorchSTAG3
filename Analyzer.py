import os
import sys
from datetime import datetime as dt
import matplotlib.pyplot as plt
import statistics
import torch
import torch.nn as nn
import torch.optim as optim
import StagNN
import AhrUtil as au
import TrainedKey as tkey


#get prediction list for single date
def predSingleDate(skNum):
	#load hyperparams
	hparams = au.getHyperparams(skNum)
	spd = int(hparams['spd'])
	tvi = int(hparams['tvi'])
	plateau = float(hparams['plateau'])
	indMask = hparams['ind_mask']
	
	#get date to make pred
	predDate = input("Enter prediction date (YYYY-MM-DD) : ")

	#determine if dates MS State matches MS Mask hparam
	matches_ms_mask = False
	msPath = os.path.join('..', 'in', 'mstates.txt')
	with open(msPath, 'r') as msFile:
		for msLine in msFile:
			lineEles = msLine.strip().split(',')
			msDate = lineEles[0]
			if msDate == predDate:
				msState = lineEles[2]
				matches_ms_mask = au.compareMasks(hparams['ms_mask'], msState)
				#print('MS Mask: ', hparams['ms_mask'], '  |  MS Itr: ', msState, '  |  Match: ', matches_ms_mask)
	if not matches_ms_mask:
		print('Market state does not match for pred date.')
		sys.exit()

	#load NN, check vals
	nnPath = os.path.join('..', 'out', 'sk', 'log', 'ann', 'structure', 'struct_'+skNum+'.pt')
	inputSize = indMask.count('1')
	hiddenSize = inputSize * 2
	outputSize = 1
	mynn = StagNN.Regressor1(inputSize, hiddenSize, outputSize)
	mynn.load_state_dict(torch.load(nnPath))

	#initialize long & short buffers : [0] ticker, [1] nn calc val, [2] actual val
	bdPath = os.path.join('..', '..', 'DB_Intrinio', 'Clean', 'ByDate', predDate+'.txt')
	longBuf = []
	shortBuf = []
	for i in range(spd):
		longLine = []
		longLine.append('PH')
		longLine.append(0.0)
		longLine.append(0.0)
		longBuf.append(longLine)
		shortLine = []
		shortLine.append('PH')
		shortLine.append(1.0)
		shortLine.append(1.0)
		shortBuf.append(shortLine)
	minIdx = 0
	minVal = 0.0
	maxIdx = 0
	maxVal = 1.0
	# itr thru Clean ByDate file
	with open(bdPath, 'r') as bdFile:
		for bdLine in bdFile:
			lineEles = bdLine.strip().split('~')
			matches_nar, nline = au.normCleanLine(lineEles, tvi, plateau, hparams['ind_mask'], hparams['nar_mask'])
			#print('--> matches_nar = ', matches_nar)
			#print('--> nline = ', nline)
			if matches_nar:
				ticker = nline[0]
				#calc output val of nn
				inData = list(map(float, nline[1:-1]))
				inTensor = torch.tensor([inData])
				with torch.no_grad():
					output = mynn(inTensor)
				outputVal = output.item()
				#print(f'Score for {ticker} : {outputVal:.7f}') 
				actualVal = 0.5
				if nline[-1] != 'tbd':
					actualVal = float(nline[-1])
				#update long buf
				if outputVal > minVal:
					longBuf[minIdx][0] = ticker
					longBuf[minIdx][1] = outputVal
					longBuf[minIdx][2] = actualVal
					minVal = 1.0
					for i in range(len(longBuf)):
						if longBuf[i][1] < minVal:
							minVal = longBuf[i][1]
							minIdx = i
				#update short buf
				if outputVal < maxVal:
					shortBuf[maxIdx][0] = ticker
					shortBuf[maxIdx][1] = outputVal
					shortBuf[maxIdx][2] = actualVal
					maxVal = 0.0
					for i in range(len(shortBuf)):
						if shortBuf[i][1] > maxVal:
							maxVal = shortBuf[i][1]
							maxIdx = i
	print('***** Long Preds for ', predDate, ' *****')
	for i in range(len(longBuf)):
		print('  ', i, ') ', longBuf[i])
	print('***** Short Preds for ', predDate, ' *****')
	for i in range(len(shortBuf)):
		print('  ', i, ') ', shortBuf[i])
	#calc avg long & short predictions in % appr
	avgPredLong = 0.0
	avgPredShort = 0.0
	for i in range(len(longBuf)):
		apprLong = (longBuf[i][2] - 0.5) * plateau
		apprShort = (shortBuf[i][2] - 0.5) * plateau
		avgPredLong += apprLong
		avgPredShort += apprShort
	avgPredLong = avgPredLong / len(longBuf)
	avgPredShort = avgPredShort / len(shortBuf)
	print('Avg Long Pred (%) : ', avgPredLong)
	print('Avg Short Pred (%) : ', avgPredShort)

#get prediction list for single date
def predSingleDate2(predDate, hparams, mynn):
	#convert non-string hyperparams
	spd = int(hparams['spd'])
	tvi = int(hparams['tvi'])
	plateau = float(hparams['plateau'])
	
	#determine if dates MS State matches MS Mask hparam
	matches_ms_mask = False
	msPath = os.path.join('..', 'in', 'mstates.txt')
	with open(msPath, 'r') as msFile:
		for msLine in msFile:
			lineEles = msLine.strip().split(',')
			msDate = lineEles[0]
			if msDate == predDate:
				msState = lineEles[2]
				matches_ms_mask = au.compareMasks(hparams['ms_mask'], msState)
				#print('MS Mask: ', hparams['ms_mask'], '  |  MS Itr: ', msState, '  |  Match: ', matches_ms_mask)
	if not matches_ms_mask:
		print('Market state does not match for pred date.')
		sys.exit()

	#initialize long & short buffers : [0] ticker, [1] nn calc val, [2] actual val
	bdPath = os.path.join('..', '..', 'DB_Intrinio', 'Clean', 'ByDate', predDate+'.txt')
	longBuf = []
	shortBuf = []
	for i in range(spd):
		longLine = []
		longLine.append('PH')
		longLine.append(0.0)
		longLine.append(0.0)
		longBuf.append(longLine)
		shortLine = []
		shortLine.append('PH')
		shortLine.append(1.0)
		shortLine.append(1.0)
		shortBuf.append(shortLine)
	minIdx = 0
	minVal = 0.0
	maxIdx = 0
	maxVal = 1.0
	# itr thru Clean ByDate file
	with open(bdPath, 'r') as bdFile:
		for bdLine in bdFile:
			lineEles = bdLine.strip().split('~')
			matches_nar, nline = au.normCleanLine(lineEles, tvi, plateau, hparams['ind_mask'], hparams['nar_mask'])
			#print('--> matches_nar = ', matches_nar)
			#print('--> nline = ', nline)
			if matches_nar:
				ticker = nline[0]
				#calc output val of nn
				inData = list(map(float, nline[1:-1]))
				inTensor = torch.tensor([inData])
				with torch.no_grad():
					output = mynn(inTensor)
				outputVal = output.item()
				#print(f'Score for {ticker} : {outputVal:.7f}') 
				actualVal = 0.5
				if nline[-1] != 'tbd':
					actualVal = float(nline[-1])
				#update long buf
				if outputVal > minVal:
					longBuf[minIdx][0] = ticker
					longBuf[minIdx][1] = outputVal
					longBuf[minIdx][2] = actualVal
					minVal = 1.0
					for i in range(len(longBuf)):
						if longBuf[i][1] < minVal:
							minVal = longBuf[i][1]
							minIdx = i
				#update short buf
				if outputVal < maxVal:
					shortBuf[maxIdx][0] = ticker
					shortBuf[maxIdx][1] = outputVal
					shortBuf[maxIdx][2] = actualVal
					maxVal = 0.0
					for i in range(len(shortBuf)):
						if shortBuf[i][1] > maxVal:
							maxVal = shortBuf[i][1]
							maxIdx = i
	#calc avg long & short predictions in % appr
	avgPredLong = 0.0
	avgPredShort = 0.0
	for i in range(len(longBuf)):
		apprLong = (longBuf[i][2] - 0.5) * plateau
		apprShort = (shortBuf[i][2] - 0.5) * plateau
		avgPredLong += apprLong
		avgPredShort += apprShort
	avgPredLong = avgPredLong / len(longBuf)
	avgPredShort = avgPredShort / len(shortBuf)
	return longBuf, avgPredLong, shortBuf, avgPredShort



#show line plot showing performance of NN over time
#out: line plots (2 on top of each other)
def predPerformance(skNum):
	#load hyperparams
	hparams = au.getHyperparams(skNum)
	#print('*** Hyperparams ***\n', hparams)
	#initalize hyperparams
	spd = int(hparams['spd'])
	tvi = int(hparams['tvi'])
	plateau = float(hparams['plateau'])
	msMask = hparams['ms_mask']
	indMask = hparams['ind_mask']
	
	#get date range
	sdate = input('Enter Start Date (YYYY-MM-DD) : ')
	edate = input('Enter End Date (YYYY-MM-DD) : ')
	#dates = au.getDatesBetween(sdate, edate)
	msDates = au.getDatesBetweenAndMS(sdate, edate, msMask)
	print('*** Dates To Apply Performance ***\n', msDates)

	#load NN, check vals
	nnPath = os.path.join('..', 'out', 'sk', 'log', 'ann', 'structure', 'struct_'+skNum+'.pt')
	inputSize = indMask.count('1')
	hiddenSize = inputSize * 2
	outputSize = 1
	mynn = StagNN.Regressor1(inputSize, hiddenSize, outputSize)
	mynn.load_state_dict(torch.load(nnPath))

	#get perf vals from every date and struct it for time-series plot
	tsDates = []
	tsLongVals = [0.0]
	tsShortVals = [0.0]
	for itrDate in msDates:
		longBuf, avgPredLong, shortBuf, avgPredShort = predSingleDate2(itrDate, hparams, mynn)
		tsDates.append(dt.strptime(itrDate, "%Y-%m-%d"))
		tsLongVals.append(tsLongVals[-1] + avgPredLong)
		tsShortVals.append(tsShortVals[-1] + avgPredShort)
		print('===== ', itrDate, ' =====')
		print('--> avgPredLong  = ', avgPredLong)
		print('--> avgPredShort = ', avgPredShort)
	tsLongVals = tsLongVals[1:]
	tsShortVals = tsShortVals[1:]
	print('--> tsLongVals  : ', tsLongVals)
	print('--> tsShortVals : ', tsShortVals)

	#plot time-series
	fig, (longPlot, shortPlot) = plt.subplots(2)
	longPlot.plot(tsDates, tsLongVals, marker = 'o')
	longPlot.set_xlabel('Date')
	longPlot.set_ylabel('APAPT')
	longPlot.set_title('Perf of Long Trades')
	longPlot.grid(True)
	shortPlot.plot(tsDates, tsShortVals, marker = 'o')
	shortPlot.set_xlabel('Date')
	shortPlot.set_ylabel('APAPT')
	shortPlot.set_title('Perf of Short Trades')
	shortPlot.grid(True)
	plt.tight_layout()
	plt.show()
		


#plot histogram plots for actual and calced TV values
#out: histogram plot (2 on top of each other)
def targetVarHist(skNum):
	#load hyperparams
	hparams = au.getHyperparams(skNum)
	#load NN
	nnPath = os.path.join('..', 'out', 'sk', 'log', 'ann', 'structure', 'struct_'+skNum+'.pt')
	inputSize = hparams['ind_mask'].count('1')
	hiddenSize = inputSize * 2
	outputSize = 1
	mynn = StagNN.Regressor1(inputSize, hiddenSize, outputSize)
	mynn.load_state_dict(torch.load(nnPath))
	#relv vars
	sdate = hparams['start_date']
	edate = hparams['end_date']
	tvi = int(hparams['tvi'])
	plateau = float(hparams['plateau'])
	msMask = hparams['ms_mask']
	indMask = hparams['ind_mask']
	narMask = hparams['nar_mask']
	dates = au.getDatesBetweenAndMS(sdate, edate, msMask)
	#get the target var values
	actualTVs = []
	calcedTVs = []
	bdBasePath = os.path.join('..', '..', 'DB_Intrinio', 'Clean', 'ByDate')
	for itrDate in dates:
		bdFullPath = os.path.join(bdBasePath, itrDate+'.txt')
		with open(bdFullPath, 'r') as bdFile:
			for bdLine in bdFile:
				lineEles = bdLine.strip().split('~')
				matches_nar, nline = au.normCleanLine(lineEles, tvi, plateau, indMask, narMask)
				if matches_nar:
					#calc output val of nn
					inData = list(map(float, nline[1:-1]))
					inTensor = torch.tensor([inData])
					with torch.no_grad():
						output = mynn(inTensor)
					calcedVal = output.item()
					#print(f'Score for {ticker} : {outputVal:.7f}') 
					actualVal = 0.5
					if nline[-1] != 'tbd':
						actualVal = float(nline[-1])				
					#add vals to lists the will show in plots
					actualTVs.append(actualVal)
					calcedTVs.append(calcedVal)
	#calc stats of pop
	actMean = statistics.mean(actualTVs)
	actStdDev = statistics.stdev(actualTVs)
	actVar = statistics.variance(actualTVs)
	print(f'--> Mean : {actMean:.5f}')
	print(f'--> Std Dev : {actStdDev:.5f}')
	print(f'--> Variance : {actVar:.5f}')
	calcMean = statistics.mean(calcedTVs)
	calcStdDev = statistics.stdev(calcedTVs)
	calcVar = statistics.variance(calcedTVs)
	print(f'--> Mean : {calcMean:.5f}')
	print(f'--> Std Dev : {calcStdDev:.5f}')
	print(f'--> Variance : {calcVar:.5f}')
	#create histogram plot
	fig, (actPlot, calcPlot) = plt.subplots(2)
	actPlot.hist(actualTVs, bins=100)
	actPlot.set_xlabel('Actual TV Value')
	actPlot.set_ylabel('Frequency')
	actPlot.set_title('Spread of TV Values')
	actPlot.set_xlim(0, 1)
	calcPlot.hist(calcedTVs, bins=100)
	calcPlot.set_xlabel('Calced TV Value')
	calcPlot.set_ylabel('Frequency')
	calcPlot.set_title('Spread of TV Values')
	calcPlot.set_xlim(0, 1)
	plt.tight_layout()
	plt.show()


#group real TVs in equal groups for binomial ANN
#out: std out
def groupRealTVs(skNum, groupNum):
	#load hyperparams
	hparams = au.getHyperparams(skNum)
	sdate = hparams['start_date']
	edate = hparams['end_date']
	tvi = int(hparams['tvi'])
	plateau = float(hparams['plateau'])
	msMask = hparams['ms_mask']
	indMask = hparams['ind_mask']
	narMask = hparams['nar_mask']
	dates = au.getDatesBetweenAndMS(sdate, edate, msMask)
	#get the target var values
	actualTVs = []
	bdBasePath = os.path.join('..', '..', 'DB_Intrinio', 'Clean', 'ByDate')
	for itrDate in dates:
		bdFullPath = os.path.join(bdBasePath, itrDate+'.txt')
		with open(bdFullPath, 'r') as bdFile:
			for bdLine in bdFile:
				lineEles = bdLine.strip().split('~')
				matches_nar, nline = au.normCleanLine(lineEles, tvi, plateau, indMask, narMask)
				if matches_nar:
					actualVal = 0.5
					if nline[-1] != 'tbd':
						actualVal = float(nline[-1])				
					#add vals to lists the will show in plots
					actualTVs.append(actualVal)
	actualTVs = sorted(actualTVs)
	stepSize = float(len(actualTVs) / groupNum)
	increment = 0.0
	print('--> actualTVs len =', len(actualTVs))
	print('Group Z [ 0 ] | value =', actualTVs[0])
	for i in range(groupNum):
		increment += stepSize
		idx = round(increment-1.0)
		print('Group', i, '[', idx, '] | value =', actualTVs[idx])



#plots recent DMC for a specific stock/date
#out : line graph (single)
def stockDMC():
	#get user input
	ticker = input('Enter Ticker : ')
	date = input('Enter Date : ')
	#init data range
	lbPeriod = 22
	dates = []
	with open(os.path.join('..', 'in', 'open_dates.txt'), 'r') as odFile:
		for odLine in odFile:
			lineEles = odLine.strip().split(',')
			dates.append(lineEles[0])
	dates.reverse()
	dateIdx = dates.index(date)
	if dateIdx > lbPeriod:
		dates = dates[dateIdx-lbPeriod:dateIdx+1]
	else:
		dates = dates[:dateIdx+1]
	#init DMCs
	dmcs = []
	for idate in dates:
		dmcs.append(0.0)
	#get DMC from SBase file
	sbPath = os.path.join('..', '..', 'DB_Intrinio', 'Main', 'S_Base', ticker+'.txt')
	with open(sbPath, 'r') as sbFile:
		for sbLine in sbFile:
			lineEles = sbLine.strip().split('~')
			sbDate = lineEles[0]
			sbDMC = float(lineEles[6]) / 1000000.0
			if sbDate in dates:
				dmcs[dates.index(sbDate)] = sbDMC
	#print median DMC val
	print('Median ', ticker, ' DMC : ', statistics.median(dmcs))
	#plot the data
	plt.plot(dates, dmcs, marker = 'o')
	plt.title('DMC Day by Day')
	plt.xlabel('Date')
	plt.ylabel('Daily Market Cap (in millions $)')
	plt.xticks(rotation = 90)
	#plt.subplots_adjust(bottom = 0.2)
	plt.grid(True)
	plt.tight_layout()
	plt.show()

#plot that show each class's TT rate over being trained	
#out : line graph (multi)
def classificationError():
	elPath = os.path.join('..', 'out', 'cls_err_log.txt')
	tlCount = []
	classTTRates = []
	classNum = 0
	is_first_line_in_file = True
	with open(elPath, 'r') as errLog:
		for elLine in errLog:
			lineEles = elLine.strip().split(',')
			if is_first_line_in_file:
				is_first_line_in_file = False
				classNum = int((len(lineEles) - 2) / 3)
				print('In classificationError(), classNum = ', classNum)
			tlCount.append(lineEles[1])
			line = []
			for i in range(classNum):
				itrTotPred = float(lineEles[2+(1+(i*3))])
				itrTotRight = float(lineEles[2+(2+(i*3))])
				itrPercent = 0.0
				if (itrTotPred != 0.0):
					itrPercent = (itrTotRight / itrTotPred) * 100.0
				line.append(itrPercent)
			classTTRates.append(line)
	print('--> tlCount : ', tlCount)
	print('--> classTTRates : ', classTTRates)
	#create plot
	xStepSize = round(len(tlCount) / 20)
	plt.plot(tlCount, classTTRates)
	plt.title('Correct Class Prediction Rates')
	plt.xlabel('Train Lines')
	plt.ylabel('Tot Right / Tot Pred (%)')
	plt.xticks(rotation = 90)
	plt.xticks(tlCount[::xStepSize], tlCount[::xStepSize])
	#plt.subplots_adjust(bottom = 0.2)
	plt.grid(True)
	plt.tight_layout()
	plt.show()


#TODO add funct that plots all stocks DMCs for a single date
def dateDMC():
	pass

#temp code, do tests here
def tempCode():
	skey = tkey.SingleKey('187')
	skey.calcKeysPerf()
	skey = tkey.SingleKey('188')
	skey.calcKeysPerf()
	skey = tkey.SingleKey('189')
	skey.calcKeysPerf()
	print('--> tempCode ... DONE')



#get user input
promptIn = """***** Analyzer Option *****
  0) Temp Code
  1) Prediction List for Single Date
  2) Performance Over Date Range
  3) Target Var Histogram
  4) Real TV Groupings
  5) Recent DMC for Single Stock & Date
  6) Classification Error Plot
Enter : """
pick = int(input(promptIn))

if pick == 0:
	tempCode()
elif pick == 1:
	skNum = (input('Enter SK Number : ')).strip()
	predSingleDate(skNum)
elif pick == 2:
	skNum = (input('Enter SK Number : ')).strip()
	predPerformance(skNum)
elif pick == 3:
	skNum = (input('Enter SK Number : ')).strip()
	targetVarHist(skNum)
elif pick == 4:
	skNum = (input('Enter SK Number : ')).strip()
	groupNum = int((input('Enter Groupings : ')).strip())
	groupRealTVs(skNum, groupNum)
elif pick == 5:
	stockDMC()
elif pick == 6:
	classificationError()
else:
	print('Invalid selection.')

