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
	dataset = dataset.reset_index(drop=True)
	featurevector = np.empty([1,42])
	idx_0 = dataset.first_valid_index()
	for idx in range(idx_0,len(dataset)+idx_0):
		trainingfeats = []
		for feature in features_:
			vec = dataset.loc[idx,feature]
			if feature == "label":
				if vec is not None:
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

	featurevector[featurevector == np.inf] = 0
	featurevector[featurevector == -np.inf] = 0
	return featurevector[1:,:]
			


def unpickle(path):
    with open(path,'rb') as file:
        data = pickle.load(file)
        file.close()
    return data


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

dataframe = "/home/fdunbar/Research/Data/MSGL_Data_100_logtransformlevels_dist5.pkl"

dataframe = pd.read_pickle(dataframe)
print(dataframe["Area"].unique())



#thwaites = ['thwaites']

features = ['Srough','contrast','dissimilarity','correlation','variance','entropy','label'] #,'Srough','label'] take out surface roughness

trainingdata = dataframe[dataframe["Area"]== 'dubawnt']

trainingdata = trainingdata.append(dataframe[dataframe["Area"] == 'brooks'])
testdata = dataframe[dataframe["Area"]=='testset']  
thwaitesdat = dataframe[dataframe["Area"] == 'thwaites']

#trainingdata[trainingdata["Area"] == 'dubawnt']["label"] = 1 # temp

print("Processing trainingdata\n")
trainingdata = unwrap(trainingdata,features)

print("Processing testdata \n")
testdata = unwrap(testdata,features)

print("Processing thwaitesdata\n")
thwaitesdata = unwrap(thwaitesdat,features)



trainingdataset = Dataset(trainingdata[:,:-1],trainingdata[:,-1])

testdataset = Dataset(testdata[:,:-1],testdata[:,-1])


thwaitesdataset = Dataset(thwaitesdata[:,:-1],thwaitesdata[:,-1])

traindataloc = "/home/fdunbar/Research/surface-stats/trainingdata_logtransformed_dist5.pkl"
testdataloc = "/home/fdunbar/Research/surface-stats/testdata_logtransformed_dist5.pkl"

writedata(trainingdataset,traindataloc)
writedata(testdataset,testdataloc)
writedata(thwaitesdataset,"/home/fdunbar/Research/surface-stats/thwaitesdata_logtrans_dist5.pkl")

