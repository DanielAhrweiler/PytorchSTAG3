import os
from datetime import datetime as dt

#get all market dates b/w 2 dates
def getDatesBetween(startDate, endDate):
	dates = []
	odPath = os.path.join('..', 'in', 'open_dates.txt')
	sdate = dt.strptime(startDate, '%Y-%m-%d')
	edate = dt.strptime(endDate, '%Y-%m-%d')
	with open(odPath, 'r') as odFile:
		for odLine in odFile:
			lineEles = odLine.strip().split(',')
			itrDate = dt.strptime(lineEles[0], '%Y-%m-%d')
			if ((itrDate >= sdate) & (itrDate <= edate)):
				dates.append(str(itrDate.date()))
	return dates

#get list of dates in range and match given MS Mask
def getDatesBetweenAndMS(sdate, edate, msMask):
	print('sdate : ', sdate)
	print('edate : ', edate)
	print('msMask : ', msMask)
	msDates = []
	msPath = os.path.join('..', 'in', 'mstates.txt')
	with open(msPath, 'r') as msFile:
		for msLine in msFile:
			lineEles = msLine.strip().split(',')
			msDate = lineEles[0]
			matches_ms_mask = False
			date_in_range = isDateInRange(msDate, sdate, edate)
			if date_in_range:
				msState = lineEles[2]
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

#get SK data from keys_struct and convert to dict obj
def getHyperparams(skNum):
	ksPath = os.path.join('..', 'out', 'sk', 'log', 'ann', 'keys_struct.txt')
	skData = []
	hparams = {}
	with open(ksPath, 'r') as ksFile:
		ksFile.readline()
		for ksLine in ksFile:
			lineEles = ksLine.strip().split(',')
			if lineEles[0] == skNum:
				skData = lineEles
				break
	if (len(skData) >= 15):
		hparams['start_date'] = skData[5]
		hparams['end_date'] = skData[6]
		hparams['call'] = skData[7]
		hparams['learn_rate'] = skData[8]
		hparams['plateau'] = skData[9]
		hparams['spd'] = skData[10]
		hparams['tvi'] = skData[11]
		hparams['ms_mask'] = skData[12]
		hparams['ind_mask'] = skData[13]
		hparams['nar_mask'] = skData[14]
	return hparams	
	
#general funct comparing mask strings
def compareMasks(baseMask, itrMask):
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
	print('--> writeToFile : ', fpath, '... DONE')


def normCleanLine(lineEles, tvi, plateau, indMask, narMask):
	#line index
	narIdx = len(indMask) + 1
	tviStartIdx = len(indMask) + 2
	#check Clean ByDate file for good NAR val
	narItr = lineEles[narIdx]
	matches_nar = compareMasks(narMask, narItr)
	nline = []
	nline.append(lineEles[0]) 
	if matches_nar:
		for c in range(len(indMask)):
			if indMask[c] == '1':
				#add feature space var
				fsVal = (1.0/65535.0) * float(lineEles[c+1])
				fsStr = f"{fsVal:.7f}"
				nline.append(fsStr)
		#add target var
		targetVal = 0.0
		if lineEles[tviStartIdx+tvi] != 'tbd':
			targetVal = float(lineEles[tviStartIdx+tvi])
		if targetVal > plateau:
			targetVal = plateau
		if targetVal < (plateau * -1.0):
			targetVal = (plateau * -1.0)
		tvRange = plateau * 2.0
		azAppr = targetVal + plateau
		tvNorm = (1.0/tvRange) * azAppr
		tvNormStr = f"{tvNorm:.7f}"
		nline.append(tvNormStr)
	return matches_nar, nline

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

