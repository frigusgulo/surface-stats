import numpy as np 
from dem import DEM,Tile,DEM_Observer
from os.path import join
import warnings
import pdb
warnings.filterwarnings('ignore')
if __name__ == "__main__":


	features = ['contrast', 'dissimilarity', 'homogeneity', 'energy','ASM','srough']

	traindatalist  = ["dubawnt_dem.npy","brooks_dem.npy"]
	observer = DEM_Observer()
	[observer.load_data(data,features=features) for data in traindatalist]
	observer.make_training_data()
	observer.save_td("training_data_matrix")
	
	'''
	thwaitesdatalist = ["thwaites_dem.npy"]
	observer = DEM_Observer()
	[observer.load_data(data,features=features) for data in thwaitesdatalist]
	observer.make_training_data()
	observer.save_td("thwaites_data_matrix")
	'''
	testdatalist = ["testdat_dem.npy"]
	observer = DEM_Observer()
	[observer.load_data(data,features=features) for data in testdatalist]
	observer.make_training_data()
	observer.save_td("test_data_matrix")


	margolddatalist = ["margold_dem.npy"]
	observer = DEM_Observer()
	[observer.load_data(data,features=features) for data in margolddatalist]
	observer.make_training_data()
	observer.save_td("margold_data_matrix")