import numpy as np 
from dem import DEM,Tile
from os.path import join

if __name__ == "__main__":
	data_dir = "/home/fdunbar/Research/Data"
	MSGL_Data = join(data_dir,"Dubawnt/MSGL_Large.npy")
	MSGL_Labels = join(data_dir,"Dubawnt/MSGL_Large_Labels.npy")

	brooks_data = join(data_dir,"Brooks/brooks_dem.npy")
	brooks_labels = join(data_dir,"Brooks/brooks_labels")

	test_data = join(data_dir,"TestSet/MSGL_testdata.npy")
	test_labels = join(data_dir,"TestSet/Testset_Labels.npy")

	thwaites_data = join(data_dir,"Thwaites/thwaitesdem.npy")
	