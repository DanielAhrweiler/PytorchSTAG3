
class FCI:
	def __init__(self):
		self.__hasHeader = False
	def __init__(self, headerBool, filePath, delim=','):
		#fill in input params
		self.__hasHeader = headerBool
		self.__filePath = filePath
		self.__columnNames = []
		#get file col infor diff ways according to if file has header
		if self.__hasHeader:
			#scan in 1st row in file
			with open(self.__filePath) as inFile:
				rawHeader = inFile.readline().strip('\n')
				if '//' in rawHeader:
					headerEles = rawHeader.split(delim)
					for ele in headerEles:
						nameOnly = ele.split(' ')[-1]
						self.__columnNames.append(nameOnly)
				else:
					print("WARNING: Actual Header? ==> ", rawHeader)
		else:
			fciLogFile = open('./../in/fci_log.txt')
			for line in fciLogFile:
				eles = line.rstrip().split(',')
				itrFilePath = eles[0]
				itrColNames = eles[1:]
				if itrFilePath in self.__filePath or itrFilePath[:-1] in self.__filePath:
					self.__columnNames = itrColNames

	def getHasHeader(self):
		return self.__hasHeader
	def getTag(self, idx):
		return self.__columnNames[idx]
	def getTags(self):
		return self.__columnNames
	def getIdx(self, tag):
		return self.__columnNames.index(tag)
	def getNumOfCols(self):
		return len(self.__columnNames)

