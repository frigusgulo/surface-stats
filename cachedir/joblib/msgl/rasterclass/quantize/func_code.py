# first line: 21
@memory.cache
def quantize(raster):
	print("\n Quantizing \n")
	if raster.detrend_: 
		raster.raster += (np.abs(np.min(raster.raster)) + 1)
		mean = np.nanmean(raster.raster[raster.raster > 0])
		std = np.nanstd(raster.raster[raster.raster > 0])

		raster.raster[raster.raster == None] = 0 # set all None values to 0
		raster.raster[np.isnan(raster.raster)] = 0
		raster.raster = np.rint(raster.raster)
		raster.raster = raster.raster.astype(int)

		raster.raster[raster.raster > (mean + 1.5*std)] = 0
		raster.raster[raster.raster < (mean - 1.5*std)] = 0 # High pass filter

		raster.raster[raster.raster > 0] = raster.raster[raster.raster > 0] - (np.min(raster.raster[raster.raster > 0]) - 1)

		raster.raster[raster.raster>101] = 0

		
		flat = np.ndarray.flatten(raster.raster[raster.raster > 0])
		range = np.max(flat) - np.min(flat)
		print("\n\nRaster Range: {}\n\n".format(range))
		#self.raster[self.raster > 101] = 0 #remove values greater than 101 for the GLCM

	else:
		raise ValueError("Raster Has Not Been Detrended")
