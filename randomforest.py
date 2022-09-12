"""
@purpose : Class for RandomForest model to initiate, train and evaluate
@when : 09/01/22
"""
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
# smoothing
from scipy.signal import savgol_filter
from statsmodels.nonparametric.kernel_regression import KernelReg

plt.style.use('seaborn-whitegrid')

class RandomForest(object):


	def __init__(self):
		# values as per C3AI, max_features = None : using all features when looking for best split
		self.rf = RandomForestRegressor(n_estimators = 500, max_depth = 5, max_features = None)

	def set_parameters(self, number_of_trees, maximum_depth):

		self.rf.n_estimators = number_of_trees
		self.rf.max_depth = maximum_depth

	# To plot the features as per the importance on a bar graph
	def get_feature_importance(self, input_features, model):
		f_i = list(zip(input_features, model.feature_importances_))
		f_i.sort(key=lambda x: x[1])
		plt.barh([x[0] for x in f_i], [x[1] for x in f_i])
		#plt.show()

	# Training phase
	def train(self, train_input, train_target, input_features):
        # clf = GridSearchCV(self.rf, self.parameters, scoring='r2', cv=3) # neg_mean_squared_error
        # clf.fit(train_input, train_target)
		self.rf.fit(train_input, train_target)
		self.get_feature_importance(input_features, self.rf)
		return self.rf


	# Eval phase on validation data - compute metrics (mdpe, mdape, r2) on this
	def eval(self, model, test_input, test_target, show_out = False, fname = "temp"):
		predictions = model.predict(test_input)
		
		if show_out:
			test_data = np.concatenate((test_input, test_target.reshape(-1, 1), predictions.reshape(-1, 1)), axis = 1)
			np.savetxt(f"{fname}_prediction.csv", test_data, delimiter = ",", fmt="%10.6f")

		errors = predictions - test_target
		mdpe = 100 * np.median(errors / test_target)        # median percentage error (MPE)
		mdape = 100 * np.median(abs(errors) / test_target)  # median absolute percentage error (MDAPE)
		r2 = r2_score(predictions, test_target)             # R2 score

		return mdpe, mdape, r2

	# Runs the training and evaluation phase
	def run(self, train_input, test_input, train_target, test_target, input_features):
		self.input_features = input_features
		best_model = self.train(train_input, train_target, input_features)
		mdpe, mdape, r2 = self.eval(best_model, test_input, test_target) 
#         print('Random forest scores:')
# 		print('MDPE = {:0.2f}%, MDAPE = {:0.2f}%, R2 = {:0.2f}'.format(mdpe, mdape, r2))
        
		return mdpe, mdape, r2, best_model
        
	# def predictConPla(self, X, n = 30):
	# 	con_pla = np.zeros((X.shape[0], 30), dtype = np.double)

	# 	for t in range(n):
	# 		x_temp = X.copy()
	# 		x_temp[ :, 7+t:] = 0
	# 		con_pla[:, t] = self.rf.predict(x_temp)

	# 	return con_pla

	def predictConPlaVector(self, model, test_data, test_data_raw):
		sns.set()
		print("\nrunning predictConPlaVector")		
		test_Id = np.unique(test_data[:, 0])
		chunksize = test_data[0, 7:].shape[0]				   # get total timestamps 
		
		#["ID", "WT", "HT", "AGE", "SEX", "OPIOD", "STUDY"]
		for patient_id in test_Id:
			patient_data              = test_data[test_data[:, 0] == patient_id]
			patient_data_raw 		  = test_data_raw[test_data_raw.loc[:, 'ID'] == patient_id]
			patient_ind               = patient_data[0, 1:7]   # patient indicators
			patient_con_pla           = 0                      # prev con pla - always start with zero
			patient_time_data         = patient_data[:, 7:]    # patient dosages for time 0 - time x-1
			patient_time_data_flatten = patient_time_data.flatten('C')
			sliding_time_win          = np.concatenate((np.zeros(chunksize), patient_time_data_flatten, np.zeros(chunksize)), axis = 0)
			patient_eval              = np.zeros(sliding_time_win.shape[0])
			# print(patient_id, patient_time_data_flatten.shape, chunksize, patient_eval.shape)

			for j in np.arange( 0, sliding_time_win.shape[0] - chunksize + 1 ):
				# print(j, j + chunksize)
				patient_eval[j] = patient_con_pla
				test_row 	 = np.zeros(chunksize + 6, dtype = np.float32)  #6
				test_row[:6] = patient_ind
				# test_row[6]  = patient_con_pla
				test_row[6:] = sliding_time_win[j : j + chunksize ]         #6
				patient_con_pla = model.predict(test_row.reshape(1, -1))

			temp = patient_data_raw.iloc[:, [0,1,2,9]]
			pdata = np.zeros((patient_eval.shape[0], temp.shape[1] + 1 ))
			pdata[:temp.shape[0], :4] = np.array(temp)
			pdata[:,  4] = patient_eval
			np.savetxt( "patient_{}_{}.csv".format(patient_id, chunksize), pdata,  delimiter = ",", fmt="%10.6f")

			fig, ax = plt.subplots(figsize=(14, 7))
			fig.tight_layout(pad=3)
			ax.plot( patient_eval, color='green', label = "actual")
			
			# savgol_smoothed  = self.smooth_savgol_filter(patient_eval)
			# convolved_3      = self.smooth_convolution(patient_eval, 3)
			# convolved_10     = self.smooth_convolution(patient_eval, 10)
			
			# ax.plot( savgol_smoothed, color = "red", label = "savgol filter")
			# ax.plot( convolved_3, color = "blue", label = "convolve 3")
			# ax.plot( convolved_10, color = "yellow", label = "convolve 10")
			# plt.legend()

			ax.set_title("patient {}".format(patient_id))
			plt.savefig("patient_{}_{}.png".format(patient_id, chunksize) )
			


	def smooth_savgol_filter(self, data):
		return savgol_filter(data, 51, 3)
			
	def smooth_convolution(self, y, box_pts):
		box = np.ones(box_pts)/box_pts
		y_smooth = np.convolve(y, box, mode='same')
		return y_smooth



	


