import numpy as np 
from dem import DEM,Tile,DEM_Observer
from os.path import join
import warnings
import matplotlib.pyplot as plt
import json
import pdb
warnings.filterwarnings('ignore')

features = ['contrast', 'dissimilarity', 'homogeneity', 'energy','ASM','correlation','srough']

def normalize(data,return_vars = True):
	m,n = data.shape
	variances = []
	new = data.copy()
	for i in range(n):
		var = np.nanvar(new[:,i],axis=0)
		mean = np.nanmean(new[:,i],axis=0)
		new[:,i] = new[:,i]- mean
		new[:,i] = new[:,i] / var
		variances.append((mean,var))
	if return_vars:
		return new,variances
	else:
		return new

def nonans(data):

	wherenans = np.sum(np.isnan(data),axis=1).astype(np.bool)

	return data[~wherenans,:]


train_data = nonans(np.load("training_data_matrix.npy",allow_pickle=True))


'''
test_dat = np.load("train_data.npy",allow_pickle=True)

thwaites_dat = np.load("thwaites_data.npy",allow_pickle=True)
margold_dat = np.load("margold_data.npy",allow_pickle=True)
'''
training_data,truevars= normalize(train_data[:,1:])

training_correllogram = np.cov(training_data.T)
#pdb.set_trace()
evecs,evals = np.linalg.eig(training_correllogram)
evals = np.real(evals)
evecs = np.real(evecs)
cutoff = np.argwhere(evecs<=.95)

evecs_sum = np.cumsum(np.real(evecs[cutoff]))/np.sum(np.real(evecs[cutoff]))

#pdb.set_trace()
'''
plt.plot(evecs_sum,'b-')

plt.xlim(0,100)
plt.title(label="PCA Cumulative Variance")
plt.ylabel("Cumulative Variance")
plt.xlabel("Principal Components")
plt.show()
plt.savefig("cum_var")
'''

 # Fix it
PC = evals[:,:50] # PC eigenvectors
pdb.set_trace
datamat =  training_data
W = datamat@PC
W = np.hstack((np.ones_like(W[:,0])[:,np.newaxis],W))
load_coefs = np.linalg.inv(W.T@W)@W.T@train_data[:,0]
pdb.set_trace()
regress_coefs = PC@load_coefs
print(regress_coefs)

regression_data = {
	"Principle Components":evals[:,:50],
	"Normalization Constants": truevars,
	"Loading Coefficients" : load_coefs,
	"Regression Coefficients": regress_coefs
}

np.save("regression_data",regression_data)
pdb.set_trace()