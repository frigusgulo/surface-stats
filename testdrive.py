from msgl.rasterclass import rasterClass
import numpy as np

if __name__ == "__main__":
	
	dataframe = "/home/fdunbar/Research/Data/MSGL_Data.pkl"
	dubdem = '/home/fdunbar/Research/Data/Dubawnt/MSGL_Large.npy'
	dublab = '/home/fdunbar/Research/Data/Dubawnt/MSGL_Large_Labels.npy'
	dubname = "dubawnt"

	testdem = '/home/fdunbar/Research/Data/TestSet/MSGL_testdata.npy'
	testlabel = '/home/fdunbar/Research/Data/TestSet/MSGL_testdata_labels.npy'
	testname = "testset"

	brookdem = '/home/fdunbar/Research/Data/Brooks/brooks_dem.npy'
	brookname = "brooks"
	brookslabel = 0

	thwaitesdem = '/home/fdunbar/Research/Data/Thwaites/thwaitsdem.npy'
	thwaitesname = "thwaites"

	dubawnt = rasterClass(dubdem,dubname,dublab)

	dubawnt.runAll(path=dataframe) 

	testset =  rasterClass(testdem,testname,testlabel,df=dataframe)
	testset.runAll()

	brooks =  rasterClass(brookdem,brookname,brookslabel,df=dataframe)
	brooks.runAll()

	thwaites =  rasterClass(thwaitesdem,thwaitesname,df=dataframe)
	thwaites.runAll()

	try:
		thwaites.dataframe.to_html("msgldata.html")
	except:
		print("To html failed")