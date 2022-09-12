"""
@purpose : Class for NeuralNetwork model to initiate, train and evaluate
@when : 25/05/22
"""
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
from torchmetrics import R2Score

class NeuralNetwork(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNetwork, self).__init__()
		
		layers = []
		layers.append(nn.Linear(input_size, hidden_size[0], dtype = torch.float64))
		layers.append(nn.ReLU())
		layers.append(nn.Dropout(p = 0.4))
		layers.append(nn.BatchNorm1d(hidden_size[0], dtype = torch.float64))
		layers.append(nn.Linear(hidden_size[0], hidden_size[1], dtype = torch.float64))
		layers.append(nn.ReLU())
		layers.append(nn.Dropout(p = 0.3))
		layers.append(nn.Linear(hidden_size[1], num_classes, dtype = torch.float64))
		layers.append(nn.ReLU())
		self.layers = nn.Sequential(*layers)
		
	def forward(self, X):
		out = self.layers(X)
		return out
    	

def trainNN(input_train, target_train, hyperparameters,  val = False, disp = False):

	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('Using device: %s'%device)

	#hyperparameters
	num_epochs 			= hyperparameters["num_epochs"]
	batch_size 			= hyperparameters["batch_size"]
	learning_rate		= hyperparameters["learning_rate"]
	learning_rate_decay = hyperparameters["learning_rate_decay"]
	reg					= hyperparameters["reg"]
	training_per 		= hyperparameters["training_per"]

	train_size 			= input_train.shape[0]
	total_iter 			= int(np.ceil(train_size / batch_size))
	
	input_train = torch.from_numpy(input_train).to(device)
	target_train = torch.from_numpy(target_train).to(device)
	val_x, val_y = None, None

	if val:
		split_index = int(training_per * train_size)
		index = np.arange(0, train_size)
		np.random.shuffle(index)
		train_index, val_index = index[:split_index] , index[split_index : ]
		train_size = train_index.shape[0]
		input_train, target_train , val_x, val_y = input_train[train_index], target_train[train_index], input_train[val_index], target_train[val_index]	
		input_train = input_train.to(device)
		target_train = target_train.to(device)
		val_x = val_x.to(device)
		val_y = val_y.to(device)

	model = NeuralNetwork(hyperparameters["input_size"], hyperparameters["hidden_size"], hyperparameters["out"]).to(device)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

	for epoch in range(num_epochs) :		
		
		for iter_ in range(total_iter):
			
			if batch_size * (iter_+1) < train_size :  
				train_batch  = input_train[  iter_ * batch_size : batch_size * (iter_+1) ]
				target_batch = target_train[ iter_ * batch_size : batch_size * (iter_+1) ]
			else:
				train_batch  = input_train[  iter_ * batch_size : ]
				target_batch = target_train[ iter_ * batch_size : ]
			
			
			out = model(train_batch)
			loss = criterion(out, target_batch)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, iter_ + 1 , total_iter, loss.item()))
				   
		if val :
			model.eval()
			with torch.no_grad():
				out = model(val_x)
				loss = criterion(out, val_y)
				print(f'Validation error : {loss.item()}')
				
		
		#shuffling each epoch
		index = np.arange(0, train_size)
		np.random.shuffle(index)
		input_train, target_train = input_train[index], target_train[index]
	
	return model, device
		
	
def evalNN(model, input_test, target_test, device):
	
	input_test  = torch.from_numpy(input_test).to(device)
	target_test = torch.from_numpy(target_test).to(device)
	
	predictions = model(input_test)
	errors = torch.subtract(predictions ,target_test)
	r2score = R2Score()

	mdpe = 100 * torch.median( torch.div(errors ,target_test) )              # median percentage error (MPE)
	mdape = 100 * torch.median(torch.abs(torch.div(errors ,target_test) ))   # median absolute percentage error (MDAPE)
	predictions = predictions.detach().numpy()                               # detach for numpy operations
	target_test = target_test.detach().numpy()
	r2 = r2_score(predictions.squeeze(), target_test)             			 # R2 score

	print("results on test")
	print(f" mdpe : {mdpe} mdape : {mdape}  r2 : {r2} \n")


