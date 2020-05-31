# first line: 12
@memory.cache
def detrend(rasterclass):
		print("\n Detrending \n")
		#perform detrending by applying a gaussian filter with a std of 200m, and detrend
		trend = gaussian_filter(rasterclass.raster,sigma=200)
		rasterclass.raster -= trend
		rasterclass.detrend_ = True
