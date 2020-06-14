import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle


def writedata(object_,path):
	with open(path,'wb') as file:
		pickle.dump(object_,file)
	file.close()

def unwrap(dataset,features_):
	featurevector = np.empty([1,26])
	idx_0 = dataset.first_valid_index()
	for idx in range(idx_0,len(dataset)+idx_0):
		trainingfeats = []
		for feature in features_:
			vec = dataset.loc[idx,feature]

			if feature == "label":
				vec = np.int(vec)
			elif isinstance(vec,np.float):
				pass
			else:
				vec = np.ndarray.flatten(vec)
			trainingfeats.append(vec)
		temp = []
		for vec in trainingfeats:
			if isinstance(vec,np.ndarray):
				for item in vec.tolist():
					temp.append(item)
			else:
				temp.append(vec)
	
		trainingfeats = np.array(temp)
		featurevector = np.vstack((featurevector,trainingfeats))


	print(featurevector[1:,:])
	return featurevector[1:,:]
			




class Dataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels):
        'Initialization'
        self.labels = labels
        self.features = features

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
     
        # Load data and get label
        X = self.features[index,:]
        Y = self.labels[index]

        return X, Y

dataframe = "/home/fdunbar/Research/Data/MSGL_Data.pkl"

dataframe = pd.read_pickle(dataframe)

trainingareas = ['dubawnt']
testareas = ['test']
#thwaites = ['thwaites']

features = ['contrast','dissimilarity','homogeneity','ASM','energy','correlation','Srough','label']

trainingdata = dataframe[dataframe["Area"]=='dubawnt']
testdata = dataframe[dataframe["Area"]=='testset']  
#thwaitesdat = dataframe[dataframe["Area"] == 'thwaites']




print("Processing trainingdata\n")
trainingdata = unwrap(trainingdata,features)

print("Processing testdata \n")
testdata = unwrap(testdata,features)

#print("Processing thwaitesdata\n")
#thwaitesdata = unwrap(thwaitesdat,features)



trainingdataset = Dataset(trainingdata[:,:-1],trainingdata[:,-1])

testdataset = Dataset(testdata[:,:-1],testdata[:,-1])


#thwaitesdataset = Dataset(thwaitesdata[:,:-1],thwaitesdata[:,-1])


writedata(trainingdataset,"/home/fdunbar/Research/surface-stats/trainingdata.pkl")
writedata(testdataset,"/home/fdunbar/Research/surface-stats/testdata.pkl")
#writedata(thwaitesdataset,"/home/fdunbar/Research/surface-stats/thwaites.pkl")