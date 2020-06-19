from msgl.rasterclass import rasterClass,detrend,quantize
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

	thwaitesdem = '/home/fdunbar/Research/Data/Thwaites/thwaitesdem.npy'
	thwaitesname = "thwaites"

	dubawnt = rasterClass(dubdem,dubname,dublab)
	dubawnt.runAll(path=dataframe) 

	print("\n\n Test Set \n\n")
	testset =  rasterClass(testdem,testname,testlabel,df=dataframe)
	testset.runAll()

	brooks =  rasterClass(brookdem,brookname,labels=brookslabel,df=dataframe)
	brooks.runAll()
	

	print("\n\n Thwaites \n\n")
	thwaites =  rasterClass(thwaitesdem,name=thwaitesname,df=dataframe)
	thwaites.runAll()

	try:
		thwaites.dataframe.to_html("/home/fdunbar/Research/Data/msgldata.html")
	except:
		print("To html failed")
	