import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

gpd.GeoDataFrame(pd.DataFrame(['p1'], columns = ['geom']),
     crs = {'init':'epsg:4326'},
     geometry = [Polygon([[-107.8742198259,61.456646442],[-108.9073061023,66.289062188],[-96.2762233084,66.7614156203],[-95.243137032,62.0181345049],[-107.8742198259,61.456646442]])]).to_file('dubawnt_.shp')