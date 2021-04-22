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
	print(len(dataset))
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


dataframe = '/home/fdunbar/Research/surface-stats/margolddataframe.pkl'

dataframe = pd.read_pickle(dataframe)
print(dataframe["Area"].unique())
datasets = dataframe["Area"].unique()

features = ['Srough','contrast','dissimilarity','correlation','variance','entropy','label']

margold_orig = unwrap(dataframe[dataframe["Area"]==datasets[0]],features)

margold_radar_10 = unwrap(dataframe[dataframe["Area"] == datasets[1]],features)


margold_radar_20 = unwrap(dataframe[dataframe["Area"] == datasets[2]],features)

margold_radar_50 = unwrap(dataframe[dataframe["Area" ]== datasets[3]],features)

margold_radar_75 = unwrap(dataframe[dataframe["Area" ]== datasets[4]],features)


margold_orig = Dataset(margold_orig[:,:-1],margold_orig[:,-1])
margold_radar_10 = Dataset(margold_radar_10[:,:-1],margold_radar_10[:,-1])
margold_radar_20 = Dataset(margold_radar_20[:,:-1],margold_radar_20[:,-1])
margold_radar_50 = Dataset(margold_radar_50[:,:-1],margold_radar_50[:,-1])
margold_radar_75 = Dataset(margold_radar_75[:,:-1],margold_radar_75[:,-1])

margold_orig_loc = '/home/fdunbar/Research/surface-stats/margold_orig_data'
margold_radar_10_loc = '/home/fdunbar/Research/surface-stats/margold_radar_10_data'
margold_radar_20_loc = '/home/fdunbar/Research/surface-stats/margold_radar_20_data'
margold_radar_50_loc = '/home/fdunbar/Research/surface-stats/margold_radar_50_data'
margold_radar_75_loc = '/home/fdunbar/Research/surface-stats/margold_radar_75_data'

writedata(margold_orig,margold_orig_loc)
writedata(margold_radar_10,margold_radar_10_loc)
writedata(margold_radar_20,margold_radar_20_loc)
writedata(margold_radar_50,margold_radar_50_loc)
writedata(margold_radar_75,margold_radar_75_loc)