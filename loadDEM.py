import os
import glob
import pandas as pd


if __name__ == '__main__':	

	dems = pd.read_csv("features.csv")
	print(dems.head())
	urls = dems['fileurl']
	for url in urls:
		url = url.replace("v3.0/2m","v3.0/10m").replace("1_2_2m","10m").replace("2_2_2m","10m").replace("2_1_2m","10m").replace("1_1_2m","10m")
		os.system('wget -r -N -nH -np -R index.html* '+ url)