import os
import sys
import subprocess
import torch
import StagNN
import AhrUtil as au
from datetime import datetime as dt


#attr and perf metric calculator for single keys
class SingleKey:
	#========== CONSTRUCTOR ==========
	def __init__(self, skNum):
		self.__id = skNum
		ksPath = os.path.join('..', 'out', 'sk', 'log', 'ann', 'keys_struct.txt')
		skData = []
		self.__hyperparams = {}
		with open(ksPath, 'r') as ksFile:
			ksFile.readline()
			for ksLine in ksFile:
				lineEles = ksLine.strip().split(',')
				if lineEles[0] == str(skNum):
					skData = lineEles
					break
		if (len(skData) >= 15):
			self.__hyperparams['start_date'] = skData[5]
			self.__hyperparams['end_date'] = skData[6]
			self.__hyperparams['call'] = skData[7]
			self.__hyperparams['learn_rate'] = skData[8]
			self.__hyperparams['plateau'] = skData[9]
			self.__hyperparams['spd'] = skData[10]
			self.__hyperparams['tvi'] = skData[11]
			self.__hyperparams['ms_mask'] = skData[12]
			self.__hyperparams['ind_mask'] = skData[13]
			self.__hyperparams['nar_mask'] = skData[14]
		else:
			print('--> Len of skData : ', len(skData))
		#load in NN
		nnPath = os.path.join('..', 'out', 'sk', 'log', 'ann', 'structure', 'struct_'+str(skNum)+'.pt')
		inputSize = self.__hyperparams['ind_mask'].count('1')
		hiddenSize = inputSize * 2
		outputSize = 1
		self.__mynn = StagNN.Regressor1(inputSize, hiddenSize, outputSize)
		self.__mynn.load_state_dict(torch.load(nnPath))

	#========== GETTERS ==========
	def getStartDate(self):
		return self.__hyperparams['start_date']
	def getEndDate(self):
		return self.__hyperparams['end_date']
	def getCall(self):
		return self.__hyperparams['call']
	def getLearnRate(self):
		return self.__hyperparams['learn_rate']
	def getPlateau(self):
		return self.__hyperparams['plateau']
	def getSPD(self):
		return self.__hyperparams['spd']
	def getTVI(self):
		return self.__hyperparams['tvi']
	def getMsMask(self):
		return self.__hyperparams['ms_mask']
	def getIndMask(self):
		return self.__hyperparams['ind_mask']
	def getNarMask(self):
		return self.__hyperparams['nar_mask']


	#========== SETTERS ==========
	def setStartDate(self, date):
		self.__hyperparams['start_date'] = date
	def setEndDate(self, date):
		self.__hyperparams['end_date'] = date
	def setCall(self, call):
		self.__hyperparams['call'] = call
	def setLearnRate(self, lrate):
		self.__hyperparams['learn_rate'] = lrate
	def setPlateau(self, plat):
		self.__hyperparams['plateau'] = plat
	def setSPD(self, spd):
		self.__hyperparams['spd'] = spd
	def setTVI(self, tvi):
		self.__hyperparams['tvi'] = tvi
	def setMsMask(self, msMask):
		self.__hyperparams['ms_mask'] = msMask
	def setIndMask(self, indMask):
		self.__hyperparams['ind_mask'] = indMask
	def setNarMask(self, narMask):
		self.__hyperparams['nar_mask'] = narMask


	#========== PERF METRIC METHODS ==========
	def singleDatePredictions(self, predDate, spd, print_predictions):
		#[0] get necessary hyperparams
		tvi = int(self.__hyperparams['tvi'])
		plateau = float(self.__hyperparams['plateau'])
		msMask = self.__hyperparams['ms_mask']
		indMask = self.__hyperparams['ind_mask']
		narMask = self.__hyperparams['nar_mask']
		
		#[1] determine if dates MS state matches MS mask
		matches_ms_mask = False
		msPath = os.path.join('..', 'in', 'mstates.txt')
		with open(msPath, 'r') as msFile:
			for msLine in msFile:
				lineEles = msLine.strip().split(',')
				msDate = lineEles[0]
				if msDate == predDate:
					msState = lineEles[2]
					matches_ms_mask = au.compareMasks(msMask, msState)
		if not matches_ms_mask:
			print('Market state does not match for pred date.')
			sys.exit()
		#[2] init long & short buffers
		bdPath = os.path.join('..', '..', 'DB_Intrinio', 'Clean', 'ByDate', predDate+'.txt')
		longBuf = []
		shortBuf = []
		for i in range(int(spd)):
			longLine = []
			longLine.append('PH')		#[0] ticker
			longLine.append(0.0)		#[1] nn calculated target val
			longLine.append(0.0)		#[2] normalized target val [0,1]
			longLine.append(0.0)		#[3] actual target val
			longBuf.append(longLine)
			shortLine = []
			shortLine.append('PH')
			shortLine.append(1.0)
			shortLine.append(1.0)
			shortLine.append(0.0)
			shortBuf.append(shortLine)
		minIdx = 0
		minVal = 0.0
		maxIdx = 0
		maxVal = 1.0
		#[3] itr thru Clean ByDate file, calc buffers along the way
		with open(bdPath, 'r') as bdFile:
			for bdLine in bdFile:
				lineEles = bdLine.strip().split('~')
				matches_nar, nline, tvActStr = au.normCleanLine(lineEles, tvi, plateau, indMask, narMask)
				if matches_nar:
					ticker = nline[0]
					#calc output of NN
					inData = list(map(float, nline[1:-1]))
					inTensor = torch.tensor([inData])
					with torch.no_grad():
						output = self.__mynn(inTensor)
					outputVal = output.item()
					#get actual target val from Clean/ByDate
					tvNorm = 0.5
					tvActual = 0.0
					if nline[-1] != 'tbd':
						tvNorm = float(nline[-1])
					if tvActStr != 'tbd':
						tvActual = float(tvActStr)
					#update long buffer
					if outputVal > minVal:
						longBuf[minIdx][0] = ticker
						longBuf[minIdx][1] = outputVal
						longBuf[minIdx][2] = tvNorm
						longBuf[minIdx][3] = tvActual
						minVal = 1.0
						for i in range(len(longBuf)):
							if longBuf[i][1] < minVal:
								minVal = longBuf[i][1]
								minIdx = i
					#update short buffer
					if outputVal < maxVal:
						shortBuf[maxIdx][0] = ticker
						shortBuf[maxIdx][1] = outputVal
						shortBuf[maxIdx][2] = tvNorm
						shortBuf[maxIdx][3] = tvActual
						maxVal = 0.0
						for i in range(len(shortBuf)):
							if shortBuf[i][1] > maxVal:
								maxVal = shortBuf[i][1]
								maxIdx = i
		#[4] sort prediction buffers
		longBuf.sort(key=lambda x: (x[1] * -1.0))
		shortBuf.sort(key=lambda x: (x[1] * 1.0))
		#[5] print prediction buffers
		if print_predictions:
			print('***** Long Preds for ', predDate, ' *****')
			for i in range(len(longBuf)):
				print('  ', i, ') ', longBuf[i])
			print('***** Short Preds for ', predDate, ' *****')
			for i in range(len(shortBuf)):
				print('  ', i, ') ', shortBuf[i])
		return longBuf, shortBuf
			


	def test(self):
		print('--> Hyperparams : ', self.__hyperparams)
		print('--> Call : ', self.__hyperparams['call'])
		print('--> MS Mask : ', self.__hyperparams['ms_mask'])

	#create and write basis file
	def createBasisFile(self):
		#[0] Date
	    #[1] SK Num
		#[2] TVT Code
		#[3] Ticker
		#[4] ANN Score
		#[5] TVI Appr

		sdate = self.__hyperparams['start_date']
		edate = self.__hyperparams['end_date']
		call = self.__hyperparams['call']
		spd = int(self.__hyperparams['spd'])
		msMask = self.__hyperparams['ms_mask']
		#[0] get dates that match SK in entire date range
		basisSDate = '2015-01-01'
		basisEDate = dt.today().strftime('%Y-%m-%d')
		msDates = au.getDatesBetweenAndMS('2015-01-01', dt.today().strftime('%Y-%m-%d'), msMask)
		#[1] itr thru msDates, calc preds for each date
		basis = []
		for i in range(len(msDates)):
			longBuf, shortBuf = self.singleDatePredictions(msDates[i], spd, False)
			if call == '0':
				for j in range(len(shortBuf)):
					basisLine = []
					basisLine.append(msDates[i])
					basisLine.append(str(self.__id))
					basisLine.append(str(au.tvtCode(msDates[i], sdate, edate)))
					basisLine.append(shortBuf[j][0])
					basisLine.append(shortBuf[j][1])
					basisLine.append(shortBuf[j][3])
					basis.append(basisLine)
			elif call == '1':
				for j in range(len(longBuf)):
					basisLine = []
					basisLine.append(msDates[i])
					basisLine.append(str(self.__id))
					basisLine.append(str(au.tvtCode(msDates[i], sdate, edate)))
					basisLine.append(longBuf[j][0])
					basisLine.append(longBuf[j][1])
					basisLine.append(longBuf[j][3])
					basis.append(basisLine)
			else:
				print('ERR: call = ', call)
		#[2] write basis data to file
		basisPath = os.path.join('..', 'out', 'sk', 'baseis', 'ann', 'ANN_'+str(self.__id)+'.txt')
		au.writeToFile(basisPath, basis, ',')
	

	#fill out placeholder values in keys_struct file
	#way too much work to recode in Python, just reuse Java code
	def calcKeysPerf(self):
		#move to correct dir
		pathToDir = os.path.join('..', 'jmain')
		os.chdir(pathToDir)
		#command to exec
		command = "./calc_sk_perf \"ANN\" "+str(self.__id)
		#use subprocess to start a linux bash command
		try:
			result = subprocess.run(command, shell=True, check=True, text=True, capture_output=False)
		except subprocess.CalledProcessError as e:
			print(f"Error: {e}")


