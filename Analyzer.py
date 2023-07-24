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


#get prediction list for single date
def predSingleDate():
	#load hyperparams
	hparams = {}
	with open(os.path.join('hparams.txt'), 'r') as hpFile:
		for hpLine in hpFile:
			lineEles = hpLine.strip().split(':')
			if len(lineEles) == 2:
				key = lineEles[0]
				value = lineEles[1]
				hparams[key] = value
	#print('*** Hyperparams ***\n', hparams)
	#convert non-string hyperparams
	tvi = int(hparams['tvi'])
	plateau = float(hparams['plateau'])
	#new params needed
	spd = 10
	
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
				print('MS Mask: ', hparams['ms_mask'], '  |  MS Itr: ', msState, '  |  Match: ', matches_ms_mask)
	if not matches_ms_mask:
		print('Market state does not match for pred date.')
		sys.exit()

	#load NN, check vals
	inputSize = 24
	hiddenSize = 48
	outputSize = 1
	mynn = StagNN.H1NN(inputSize, hiddenSize, outputSize)
	mynn.load_state_dict(torch.load('mynn.pt'))

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
					shortBuf[minIdx][0] = ticker
					shortBuf[minIdx][1] = outputVal
					shortBuf[minIdx][2] = actualVal
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
	tvi = int(hparams['tvi'])
	plateau = float(hparams['plateau'])
	#new params needed
	spd = 10
	
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
				print('MS Mask: ', hparams['ms_mask'], '  |  MS Itr: ', msState, '  |  Match: ', matches_ms_mask)
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
					shortBuf[minIdx][0] = ticker
					shortBuf[minIdx][1] = outputVal
					shortBuf[minIdx][2] = actualVal
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
def predPerformance():
	#load hyperparams
	hparams = {}
	with open(os.path.join('hparams.txt'), 'r') as hpFile:
		for hpLine in hpFile:
			lineEles = hpLine.strip().split(':')
			if len(lineEles) == 2:
				key = lineEles[0]
				value = lineEles[1]
				hparams[key] = value
	#print('*** Hyperparams ***\n', hparams)
	#convert non-string hyperparams
	tvi = int(hparams['tvi'])
	plateau = float(hparams['plateau'])
	#new params needed
	spd = 10
	
	#get date range
	sdate = input('Enter Start Date (YYYY-MM-DD) : ')
	edate = input('Enter End Date (YYYY-MM-DD) : ')
	dates = au.getDatesBetween(sdate, edate)

	#determine if dates MS State matches MS Mask hparam
	msDates = []
	matches_ms_mask = False
	msPath = os.path.join('..', 'in', 'mstates.txt')
	with open(msPath, 'r') as msFile:
		for msLine in msFile:
			lineEles = msLine.strip().split(',')
			msDate = lineEles[0]
			date_in_range = au.isDateInRange(msDate, sdate, edate)
			if date_in_range:
				msState = lineEles[2]
				matches_ms_mask = au.compareMasks(hparams['ms_mask'], msState)
				print('MS Mask: ', hparams['ms_mask'], '  |  MS Itr: ', msState, '  |  Match: ', matches_ms_mask)
				if matches_ms_mask:
					msDates.append(msDate)
	msDates.reverse()
	print('*** Dates To Apply Performance ***\n', msDates)

	#load NN, check vals
	inputSize = 24
	hiddenSize = 48
	outputSize = 1
	mynn = StagNN.H1NN(inputSize, hiddenSize, outputSize)
	mynn.load_state_dict(torch.load('mynn.pt'))

	#get perf vals from every date and struct it for time-series plot
	tsDates = []
	tsLongVals = [0.0]
	tsShortVals = [0.0]
	for itrDate in msDates:
		longBuf, avgPredLong, shortBuf, avgPredShort = predSingleDate2(itrDate, hparams, mynn)
		tsDates.append(dt.strptime(itrDate, "%Y-%m-%d"))
		tsLongVals.append(tsLongVals[-1] + avgPredLong)
		tsShortVals.append(tsShortVals[-1] + avgPredShort)

	#plot time-series
	fig, (longPlot, shortPlot) = plt.subplots(2)
	longPlot.plot(tsDates, tsLongVals[1:], marker = 'o')
	longPlot.set_xlabel('Date')
	longPlot.set_ylabel('APAPT')
	longPlot.set_title('Perf of Long Trades')
	longPlot.grid(True)
	shortPlot.plot(tsDates, tsShortVals[1:], marker = 'o')
	shortPlot.set_xlabel('Date')
	shortPlot.set_ylabel('APAPT')
	shortPlot.set_title('Perf of Short Trades')
	shortPlot.grid(True)
	plt.tight_layout()
	plt.show()
		


#plot histogram plots for actual and calced TV values
def targetVarHist():
	#load NN
	inputSize = 24
	hiddenSize = 48
	outputSize = 1
	mynn = StagNN.H1NN(inputSize, hiddenSize, outputSize)
	mynn.load_state_dict(torch.load('mynn.pt'))
	#relv vars
	sdate = '2016-01-01'
	edate = '2020-12-31'
	tvi = 0
	plateau = 15.0
	indMask = '111111111111111111111111'
	narMask = '1111'
	dates = au.getDatesBetween(sdate, edate)
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

#TODO add funct that plots recent DMC for a specific stock/date
def recentDMC():
	pass


#get user input
promptIn = """***** Analyzer Option *****
  1) Prediction List for Single Date
  2) Performance Over Date Range
  3) Target Var Histogram
  4) Recent DMC for Single Stock & Date
Enter : """
pick = int(input(promptIn))

picks = [predSingleDate]
picks.append(predPerformance)
picks.append(targetVarHist)
picks.append(recentDMC)
picks[pick-1]()
