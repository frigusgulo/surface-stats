# class prediction from logistic regression coefficients


import numpy as np 
from dem import DEM,Tile,DEM_Observer
from os.path import join
import warnings
import pdb


def predict(datamat,coefficients,normvars):
	m,n = datamat.shape
	labels = datamat[:,0]
	mu,sigma = normvars[:,0],normvars[:,1]

	datamat_normed = np.zeros_like(datamat[:,1:])
	for i in range(1,n-1):
		datamat_normed[:,i] = (datamat[:,i]-mu[i])/sigma[i]
	predictions = np.zeros_like(datamat[:,0])
	for j in range(m):
		predictions[j] = datamat_normed[j,:]@coefficients # log likelihoods

	predictions = 1/(1+np.exp(-1*predictions))
	return predictions,labels

def accuracy(predictions,labels):
	assert predictions.shape == labels.shape

	pdb.set_trace()

	predictions = np.rint(predictions).astype(np.uint8)
	labels = np.rint(labels).astype(np.uint8)

	tp = max(np.argwhere(predictions[labels==1]==1).shape)
	fp = max(np.argwhere(predictions[labels==0]==1).shape)
	fn = max(np.argwhere(predictions[labels==1]==0).shape)
	acc = np.rint((np.sum(predictions==labels)/max(predictions.shape)))*100

	precision = tp/(tp + fp)
	recall = tp/(tp + fn)
	print(f"Accuracy: {acc} %,Precision: {precision},Recall: {recall}\n")
	pdb.set_trace()
	return acc,precision,recall



if __name__ == "__main__":
	regression_data = np.load("regression_data.npy",allow_pickle=True)[np.newaxis][0]

	test_data = np.load("test_data_matrix.npy",allow_pickle=True)

	margold_data = np.load("margold_data_matrix.npy",allow_pickle=True)

	
	coefficients = regression_data["Regression Coefficients"]
	normconst = np.stack(regression_data["Normalization Constants"])
	
	test_data_preds,test_data_labels = predict(test_data,coefficients,normconst)

	margold_data_preds,_ = predict(margold_data,coefficients,normconst)

	testacc,testprec,testrecall = accuracy(test_data_preds,test_data_labels)

