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

#get bool if date is b/w 2 other dates
def isDateInRange(inDate, startDate, endDate):
	idate = dt.strptime(inDate, "%Y-%m-%d")
	sdate = dt.strptime(startDate, "%Y-%m-%d")
	edate = dt.strptime(endDate, "%Y-%m-%d")
	return (sdate <= idate <= edate)
	
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


