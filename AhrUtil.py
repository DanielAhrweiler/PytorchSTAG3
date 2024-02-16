import os
import torch
from datetime import datetime as dt
from FCI import FCI

#get all market dates b/w 2 dates
def getDatesBetween(startDate, endDate):
	dates = []
	odPath = os.path.join('.', '..', 'in', 'open_dates.txt')
	fciOD = FCI(False, odPath)
	sdate = dt.strptime(startDate, '%Y-%m-%d')
	edate = dt.strptime(endDate, '%Y-%m-%d')
	with open(odPath, 'r') as odFile:
		for odLine in odFile:
			lineEles = odLine.strip().split(',')
			itrDate = dt.strptime(lineEles[fciOD.getIdx('date')], '%Y-%m-%d')
			if ((itrDate >= sdate) & (itrDate <= edate)):
				dates.append(str(itrDate.date()))
	return dates

#get list of dates in range and match given MS Mask
def getDatesBetweenAndMS(sdate, edate, msMask):
	print('sdate : ', sdate)
	print('edate : ', edate)
	print('msMask : ', msMask)
	msDates = []
	msPath = os.path.join('.', '..', 'in', 'mstates.txt')
	fciMS = FCI(False, msPath)
	with open(msPath, 'r') as msFile:
		for msLine in msFile:
			lineEles = msLine.strip().split(',')
			msDate = lineEles[fciMS.getIdx('date')]
			matches_ms_mask = False
			date_in_range = isDateInRange(msDate, sdate, edate)
			if date_in_range:
				msState = lineEles[fciMS.getIdx('ms_mask')]
				matches_ms_mask = compareMasks(msMask, msState)
				#print('MS Mask: ', msMask, '  |  MS Itr: ', msState, '  |  Match: ', matches_ms_mask)
				if matches_ms_mask:
					msDates.append(msDate)
	msDates.reverse()
	return msDates

#get bool if date is b/w 2 other dates
def isDateInRange(inDate, startDate, endDate):
	idate = dt.strptime(inDate, "%Y-%m-%d")
	sdate = dt.strptime(startDate, "%Y-%m-%d")
	edate = dt.strptime(endDate, "%Y-%m-%d")
	return (sdate <= idate <= edate)

#get TVT (train, validation, test) code for basis file
def tvtCode(inDate, startDate, endDate):
	code = 0
	idate = dt.strptime(inDate, "%Y-%m-%d")
	sdate = dt.strptime(startDate, "%Y-%m-%d")
	edate = dt.strptime(endDate, "%Y-%m-%d")
	if (sdate <= idate <= edate):
		if (idate.day % 2 == 1):
			code = 1
	else:
		code = 2
	return code

#get SK data from keys_struct and convert to dict obj
def getHyperparams(skNum):
	ksPath = os.path.join('.', '..', 'out', 'sk', 'log', 'ann', 'keys_struct.txt')
	fciKS = FCI(True, ksPath)
	skData = []
	hparams = {}
	with open(ksPath, 'r') as ksFile:
		ksFile.readline()
		for ksLine in ksFile:
			lineEles = ksLine.strip().split(',')
			if lineEles[fciKS.getIdx('sk_num')] == skNum:
				skData = lineEles
				break
	if (len(skData) >= 15):
		hparams['start_date'] = skData[fciKS.getIdx('start_date')]
		hparams['end_date'] = skData[fciKS.getIdx('end_date')]
		hparams['call'] = skData[fciKS.getIdx('call')]
		hparams['learn_rate'] = skData[fciKS.getIdx('learn_rate')]
		hparams['plateau'] = skData[fciKS.getIdx('plateau')]
		hparams['spd'] = skData[fciKS.getIdx('spd')]
		hparams['tvi'] = skData[fciKS.getIdx('tvi')]
		hparams['ms_mask'] = skData[fciKS.getIdx('ms_mask')]
		hparams['ind_mask'] = skData[fciKS.getIdx('ind_mask')]
		hparams['nar_mask'] = skData[fciKS.getIdx('nar_mask')]
	return hparams	
	
#general funct comparing mask strings
def compareMasks(baseMask, itrMask):
	#compare, base can have 'x', itr cannot
	is_match = True
	for c in range(len(itrMask)):
		if (baseMask[c] != 'x'):
			if (baseMask[c] != itrMask[c]):
				is_match = False
	return is_match

#writes a 2D list to file
def writeToFile(fpath, data, delim):
	with open(fpath, 'w') as toFile:
		for x in range(len(data)):
			csvRow = delim.join(map(str, data[x]))
			if x != (len(data)-1):
				toFile.write(csvRow + '\n')
			else:
				toFile.write(csvRow)

#converts a regression TV tensor (cont vals) into a classification TV tensor (binned)
def binTargetTensor(regTT, inThresholds):
	thresholds = inThresholds.copy()
	thresholds.insert(0, 0.0)
	thresholds.append(1.01)
	clsTT = torch.zeros_like(regTT)
	for batch in range(len(regTT)):
		for row in range(len(regTT[batch])):
			itrVal = regTT[batch,row,0]
			for x in range(len(thresholds)-1):
				if (itrVal >= thresholds[x]) and (itrVal < thresholds[x+1]):
					clsTT[batch,row,0] = x
					break
	return clsTT

#calcs whether pred is right/wrong from softmax form
def isRightClassificationPrediction(smTensor, rightBin):
	maxIdx = smTensor.argmax(dim=0)
	is_right_pred = False
	if (maxIdx.item() == rightBin):
		is_right_pred = True
	return maxIdx, is_right_pred

#converts ByDate line into ML line, also returns if line matches a NAR mask
def normCleanLine(lineEles, tvi, plateau, indMask, narMask):
	#create a FCI for ByDate file
	bdPath = os.path.join('.', '..', '..', 'DB_Intrinio', 'Clean', 'ByDate')
	fciBD = FCI(False, bdPath)
	#get index for target var
	tviStartIdx = fciBD.getIdx('appr_intra1')
	#check Clean ByDate file for good NAR val
	narItr = lineEles[fciBD.getIdx('nar_mask')]
	while (len(narItr) < len(narMask)):
		narItr = narItr + 'x'
	matches_nar = compareMasks(narMask, narItr)
	nline = []
	nline.append(lineEles[fciBD.getIdx('ticker')]) 
	tvActStr = ''
	if matches_nar:
		for c in range(len(indMask)):
			indName = 'ind' + str(c)
			if indMask[c] == '1':
				#add feature space var
				fsVal = (1.0/65535.0) * float(lineEles[fciBD.getIdx(indName)])
				fsStr = f"{fsVal:.7f}"
				nline.append(fsStr)
		#add target var
		targetVal = 0.0
		tvActStr = 'tbd'
		if lineEles[tviStartIdx+tvi] != 'tbd':
			targetVal = float(lineEles[tviStartIdx+tvi])
			tvActStr = f"{targetVal:.4f}"
		if targetVal > plateau:
			targetVal = plateau
		if targetVal < (plateau * -1.0):
			targetVal = (plateau * -1.0)
		tvRange = plateau * 2.0
		azAppr = targetVal + plateau
		tvNorm = (1.0/tvRange) * azAppr
		tvNormStr = f"{tvNorm:.7f}"
		nline.append(tvNormStr)
	return matches_nar, nline, tvActStr

#analyze an obj by itr thru its dir() eles
def inDepthDir(objName, obj):
	print('======= ', objName, ' =======')
	eleList = dir(obj)
	for i in range(len(eleList)):
		if hasattr(obj, eleList[i]):
			if callable(getattr(obj, eleList[i])):
				print(f'    {i}) {eleList[i]} is METHOD')
			else:
				print(f'    {i}) {eleList[i]} is ATTR')
	print('=========================')

