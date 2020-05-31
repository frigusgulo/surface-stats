from msgl.rasterclass import rasterClass
import numpy as np

if __name__ == "__main__":
	
	dataframe = "/home/dunbar/DEM/MSGL_Data.pkl"
	dubdem = '/home/dunbar/DEM/MSGL_Large.npy'
	dublab = '/home/dunbar/DEM/MSGL_Large_Labels.npy'
	dubname = "dubawnt"

	testdem = '/home/dunbar/DEM/MSGL_testdata.npy'
	testlabel = '/home/dunbar/DEM/MSGL_testdata_labels.npy'
	testname = "testset"

	brookdem = '/home/dunbar/DEM/brooks_dem.npy'
	brookname = "brooks"
	brookslabel = 0

	thwaitesdem = '/home/dunbar/DEM/thwaitsdem.npy'
	thwaitesname = "thwaites"

	dubawnt = rasterClass(dubdem,dubname,dublab)

	dubawnt.runAll(path=dataframe) 

	testset =  rasterClass(testdem,testname,testlabel,df=dataframe)
	testset.runAll()

	brooks =  rasterClass(brookdem,brookname,brookslabel,df=dataframe)
	brooks.runAll()

	thwaites =  rasterClass(thwaitesdem,thwaitesname,df=dataframe)
	thwaites.runAll()
