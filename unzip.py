import os
import glob

if __name__ == '__main__':	

	for root,dirs,files in os.walk("elev"):
			for file in files:
				print(root,file)
				file_loc = os.path.join(root,file)
				os.system("tar -zxvf " + file_loc)