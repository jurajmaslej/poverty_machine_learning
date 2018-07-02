import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class Loader:
	
	def __init__(self, train_file = "train_data.txt"):
		train_f = np.genfromtxt(train_file, delimiter= ',', dtype = str, skip_header = 1)
		self.header = 'CensusId,State,County,TotalPop,Men,Women,Hispanic,White,Black,Native,Asian,Pacific,Citizen,Income,IncomeErr,IncomePerCap,IncomePerCapErr,Poverty,ChildPoverty,Professional,Service,Office,Construction,Production,Drive,Carpool,Transit,Walk,OtherTransp,WorkAtHome,MeanCommute,Employed,PrivateWork,PublicWork,SelfEmployed,FamilyWork,Unemployment'.split(',')
		#print(self.header)
		#shuffle data
		np.random.shuffle(train_f)
		#/shuffle data
		for i in range(0,len(train_f)):
			for j in range(0,len(train_f[i])):
				if j!=1 and j!=2:	#except state and county
					try:
						tmp = float(train_f[i,j])
					except:
						train_f[i,j] = float(0)
		
		estimation, validation = self.eighty_twenty_split(train_f)
		
		self.estimation_data = estimation
		self.validation_data = validation
		
		print(self.estimation_data.shape)
		print(self.validation_data.shape)
		self.estimation_data = self.strings_to_floats(self.estimation_data)
		self.validation_data = self.strings_to_floats(self.validation_data)
		ind1 = 0
		ind2 = 0
		# troublesome indices in original data, no clue why, manual change does not help
		#self.estimation_data[548,18] = float(0)
		#self.estimation_data[549,18] = float(0)
		#self.validation_data[97,13] = float(0)
		#self.validation_data[97,14] = float(0)
		
		np.random.shuffle(self.estimation_data)
		np.random.shuffle(self.validation_data)
		self.estimation_data = self.estimation_data.astype('float')
		
		self.validation_data = self.validation_data.astype('float')
		
	def strings_to_floats(self, data):
		encoder_state = preprocessing.LabelEncoder()
		encoder_state.fit(data[:,1])
		state_encoded = encoder_state.transform(data[:,1])
		
		encoder_county = preprocessing.LabelEncoder()
		encoder_county.fit(data[:,2])
		county_encoded = encoder_county.transform(data[:,2])
		
		data[:,1] = state_encoded
		data[:,2] = county_encoded
		return data
		
	def normalize_data(self):
		self.estimation_data -=  np.mean(self.estimation_data, axis = 1, keepdims = True)
		self.estimation_data /= np.std(self.estimation_data, axis = 1, keepdims = True)
		
	def eighty_twenty_split(self, data):
		splitter = int(data.shape[0]*0.8)
		return data[:splitter],data[splitter:]
