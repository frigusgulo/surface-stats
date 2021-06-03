import numpy as np 
from dem import DEM,Tile,DEM_Observer
from os.path import join
import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":
	
	data_dir = "../../data"
	MSGL_Data = join(data_dir,"MSGL_Large.npy")
	MSGL_Labels = join(data_dir,"MSGL_Large_Labels.npy")

	brooks_data = join(data_dir,"brooks_dem.npy")
	brooks_labels = join(data_dir,"Brooks_labels.npy")

	test_data = join(data_dir,"MSGL_testdata.npy")
	test_labels = join(data_dir,"MSGL_testdata_labels.npy")

	thwaites_data = join(data_dir,"thwaitesdem.npy")


	dubawnt = DEM("Dubawnt",MSGL_Data,MSGL_Labels,resolution=10)

	dubawnt.run("dubawnt_dem")



	brooks = DEM("Brooks",brooks_data,brooks_labels,resolution=10)

	brooks.run("brooks_dem")


	

	testdata = DEM("testset",test_data,test_labels,resolution=10)

	testdata.run("testdat_dem")





	margold = DEM("margold",join(data_dir,"margold_orig.npy"),resolution=10)

	margold.run("margold_dem")

	
	thwaites = DEM("thwaites",thwaites_data,resolution=10)

	thwaites.run("thwaites_dem")

