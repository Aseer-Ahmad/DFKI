"""
@purpose : Class for SVR model to initiate, train and evaluate
@when : 09/01/22
"""
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
import numpy as np
from sklearn.pipeline import make_pipeline

class SVM_C(object):


	def __init__(self):
		# values as per C3AI, max_features = None : using all features when looking for best split
		self.ss = StandardScaler()
		self.svm = SVR(C=1.0, epsilon=0.2)

	def set_parameters(self, kernel, degree, gamma = 'scale', C = 1.0, epsilon = 0.1 ):

		self.svm.kernel = kernel
		self.svm.degree = degree
		self.svm.gamma = gamma
		self.svm.C = C
		self.svm.epsilon = epsilon
		
	# Training phase
	def train(self, train_input, train_target, input_features):
		self.ss.fit(train_input)
		train_input = self.ss.transform(train_input)
		self.svm.fit(train_input, train_target)
		return self.svm

	# Eval phase on validation data - compute metrics (mdpe, mdape, r2) on this
	def eval(self, model, test_input, test_target):
		test_input =  self.ss.transform(test_input)
		predictions = model.predict(test_input)
		errors = predictions - test_target
		mdpe = 100 * np.median(errors / test_target)        # median percentage error (MPE)
		mdape = 100 * np.median(abs(errors) / test_target)  # median absolute percentage error (MDAPE)
		r2 = r2_score(predictions, test_target)             # R2 score

		return mdpe, mdape, r2

	# Runs the training and evaluation phase
	def run(self, train_input, test_input, train_target, test_target, input_features):
		best_model = self.train(train_input, train_target, input_features)
		mdpe, mdape, r2 = self.eval(best_model, test_input, test_target) 

		return mdpe, mdape, r2
        
	def predictConPla(self, X, n = 30):
		con_pla = np.zeros((X.shape[0], 30), dtype = np.double)

		for t in range(n):
			x_temp = X.copy()
			x_temp[ :, 7+t:] = 0
			con_pla[:, t] = self.svm.predict(x_temp)

		return con_pla


