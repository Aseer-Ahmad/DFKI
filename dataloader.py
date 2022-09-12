"""
@file : dataloader.py
@purpose : Fetch raw data, preprocess and split to train and test datasets
@note : get_data is the main method running this file
@when : 09/01/22
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import random
import itertools

# import local files here
import utils

# macros used to refer to the columns of the raw data
COL_ID = 0
COL_TIME = 1

# COL_CON_PLA = 2
# COL_MG_MIN = 3
# COL_WT = 4
# COL_HT = 5
# COL_AGE = 6
# COL_SEX = 7

COL_CON_PLA = 9
COL_MG_MIN = 2
COL_WT = 3
COL_HT = 4
COL_AGE = 5
COL_SEX = 6

#new features
COL_OP = 7
COL_STUD = 8

#new features V2
COL_CMT = 9 
COL_MDV = 10
COL_EVID = 11


chunksize = 25.

'''
Brief :     Wherever CON_PLA is not null in the raw data, there is a duplicate entry with the same time stamp.  
			This method copies the CON_PLA value to the previous row and removes this duplicate entry in order to avoid training biases
Inputs :    data : Raw data
Returns :   data : preprcessed data with duplicate entries removed 
'''
def process_duplicates(data):
	N = data.shape[0]                                           # total number of entries in raw data
	data = np.c_[data, np.zeros(N)]                             # add extra column to flag the duplicate rows which need to be removed

	for row in range(N - 1):
		# if ID and TIME are identical in two consecutive rows, its a duplicate entry
		if data[row][COL_ID] == data[row + 1][COL_ID] and data[row][COL_TIME] == data[row + 1][COL_TIME]:
			data[row][COL_CON_PLA] = data[row + 1][COL_CON_PLA] # copy the CON_PLA to the former duplicate row
			data[row + 1][-1] = 1                               # flag the later duplicate row as it needs to be removed

	data = np.delete(data, np.where(data[:, -1] == 1), axis=0)  # delete all thw rows with flag=1 in the last row
	data = np.delete(data, -1, axis=1)                          # delete the newly added flag column
	return data

'''
Brief :     This method fetches the last CON_PLA value and populates it to a new column besides the current CON_PLA value
			Note : It is done on per-patient basis !
Inputs :    data : Raw data
Returns :   data : preprocessed data with additional column containing previous CON_PLA 
			not_null_indices.tolist() : Indices of raw data where CON_PLA is not null
'''
def fill_prev_conpla(data):
	not_null_indices = np.where(~np.isnan(data[:, COL_CON_PLA]))[0]   # fetch row indices with non null CON_PLA values
	data = np.c_[data, np.zeros(data.shape[0])]                       # add a new columnW with all entries as 0

	for idx in range(1, len(not_null_indices)):
		prev = not_null_indices[idx-1]
		current = not_null_indices[idx]
		if data[current, COL_CON_PLA] == 0:                            # if current non null CON_PLA = 0, PREV_CON_PLA = 0
			data[current, -1] = 0
		else :
			data[current, -1] = data[prev, COL_CON_PLA]
	return data, not_null_indices.tolist()

'''
Brief   :   This method extracts out mg_min between two consecutive CON_PLA so as to use it as rows instead of column values 
			in the final preproc data. 
Input   :   propofol_array : raw data in numpy array form
			not_null_indices : List of indices in raw data array where CON_PLA is not null	
			patient_first_row : List of first row indices of each patient		
Returns :   all_mgmin_array : Nested list of mg_min between two consecutive CON_PLA
			max_gap : Maximum time lag between two consecutive measured CON_PLA		
			num_mg_min : length of total mg_min measured per row	
'''
def extract_mgmin_per_conpla(propofol_array, not_null_indices, patient_first_row):
	all_mgmin = []
	max_ids_list = []
	max_gap = 0
	tot_mg_min = []
    
	# begin_dx and end_idx - non null CON_PLA indices between which mg_ming needs to be extracted
	for i in range(0, len(not_null_indices) - 1):
		end_idx = not_null_indices[i + 1]
		# if CON_PLA = 0, start mg_min extraction from the same row onwards
		if propofol_array[not_null_indices[i], COL_CON_PLA] == 0:
			begin_idx = not_null_indices[i]
		# if CON_PLA is not zero, current row mg_min is for prev batch. Thus, start mg_min extraction from next row onwards
		else:
			begin_idx = not_null_indices[i] + 1

		# Skip the last mg_min for which we do not have CON_PLA
		if end_idx not in patient_first_row:
			between_mgmin = propofol_array[begin_idx:end_idx + 1, COL_MG_MIN].tolist()

			if len(between_mgmin) > max_gap:
				max_gap = len(between_mgmin)
				max_idx = end_idx
				max_ids_list.append(end_idx)

			all_mgmin.append(between_mgmin)
	print("Max gap between two measured CON_PLA: ", max_gap, " at index:", max_idx)
	# print("Gap more than 30minutes at indices : ", max_ids_list)

	'''
	Each row in all_mgmin represents the mg_min for measured CON_PLA.
	Num of mg_min for each such CON_PLA might not be the same
	Max num of mg_min for a measured CON_PLA is 68 (max_gap)
	Balance each row of all_mgmin with same num of mg_min entries by appending 0s
	'''
	for row in all_mgmin:
	    tot_mg_min.append(len(row))
	    while len(row) < max_gap:
	        row.append(0)
	all_mgmin_array = np.array(all_mgmin)

	return all_mgmin_array, max_gap, tot_mg_min


# def checkPlot(data):
# 	data_p = data[data.iloc[:, 8] == 0.0]

# 	t_data = data_p.iloc[:, 10:]
# 	data_p["temp"] = t_data.apply(ret_t, axis = 1)
# 	l = list(data_p["temp"])
# 	# print(data_p[data_p.temp > 20].iloc[:, :15])
	
# 	sns.distplot( l , hist=True, kde=True, color = 'darkblue', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
# 	plt.xlabel("first dosage gap")
# 	plt.ylabel("frequency")
# 	plt.savefig("a.jpg")
# 	plt.show()

'''
Brief   : returns indexes from the transformed data of the rows which correspond to first dosage and total gap length 
		  is below threshold
Returns : list()
'''
def getFirstDosageIndex(data, TOT_MG_MIN_INDX, size, tolerance):
	b_tol, u_tol = size - tolerance , size + tolerance
	PREV_CON_PLA_INDX = TOT_MG_MIN_INDX - 1
	indx = data[(data.iloc[: , PREV_CON_PLA_INDX] == 0) & (data.iloc[: , TOT_MG_MIN_INDX] < b_tol )].index.values.tolist()
	# print(data.iloc[indx, :15])
	return indx

'''
Brief   : Main method to start the preprocessing of the raw data
			1. Remove duplicate entries (rows) where CON_PLA is not null
			2. Add new column - PREV_CON_PLA 
			3. Extract mg_min between two consecutive CON_PLA and transpose the data
			4. Perform running cummulative sum of mg_min
Input   : Path to the raw data
Returns : Preprocessed data - trainingdata.xlsx
'''
def preprocess_data(path, prepare_test = False):
	print("in preprocess_data")
	# float precision added bcz pandas was changing the decimal values
	propofol_df = pd.read_csv(path, float_precision='round_trip')
	
	# prepare test data by study
	if not prepare_test:
		random.seed(1) # 1,6,8,9
		tf_data_shape = propofol_df.shape
		print( "transformed data shape : {}".format(tf_data_shape))
		test_per       = 0.13
		total_studies  = 4
		total_patients = len(list(propofol_df.ID.unique()))
		test_patients_per_study = int((test_per * total_patients) / total_studies)
		test_patient_ids = [random.sample( list(propofol_df[propofol_df.Study == i].ID.unique()), test_patients_per_study) for i in range(1,total_studies+1)]
		print("patient data for test from studies[1-4]: {}".format(test_patient_ids)) 
		test_patient_ids = list(itertools.chain(*test_patient_ids))
		propofol_df_test = propofol_df[propofol_df.ID.isin(test_patient_ids)]
		propofol_df = propofol_df[~propofol_df.ID.isin(test_patient_ids)]
		print("test data percentage to use for entire vector prediction : {}".format(propofol_df_test.shape[0]/tf_data_shape[0]))
		propofol_df_test.to_csv("test_data_by_study.csv", index = False)

	# Convert from dataframe to 2d array
	propofol_array = np.asarray(propofol_df, dtype = np.float32)
	propofol_array = process_duplicates(propofol_array)         # remove the duplicate row entries when CON_PLA is measured

	patient_ids = np.unique(propofol_array[:,0]).astype(int)    # fetch the patient ids
	patient_first_row = []                                      # first row of each patient in the data
	patient_rows = []
	for id in patient_ids:
		patient_first_row.append(np.where(propofol_array[:, 0] == id)[0][0].tolist())
		patient_rows.append(np.where(propofol_array[:, 0] == id)[0].tolist())

	# Populate 0 in the first row of each patient. Helps in using 0 as the PREV_CON_PLA for the first CON_PLA
	for i in patient_first_row:
		propofol_array[i][COL_CON_PLA] = 0
	
	# Add a new column to the data, fill PREV_CON_PLA in the rows with non null CON_PLA, remaining rows with value 0
	propofol_array, not_null_indices = fill_prev_conpla(propofol_array)

	input_data_df = pd.DataFrame(propofol_array)
	# Filter out rows with non-null CON_PLA
	input_data_df = input_data_df[~input_data_df[COL_CON_PLA].isna()]
	# But do not consider rows with CON_PLA=0 as it was manually added for preproc
	input_data_df = input_data_df[input_data_df[COL_CON_PLA] != 0]

	# Drop columns - time [1] and mg_min [3]
	input_data_df = input_data_df.drop(input_data_df.columns[[COL_TIME,COL_MG_MIN]], 1).reset_index(drop=True)
	# input_data_df.to_excel(utils.data_path + 'input_data_df.xlsx')      # kept to cross verify, can be removed
	
	# Extract mg_min between two consecutive measured CON_PLA and transpose the data
	all_mgmin_array, max_num_mgmin, tot_mg_min = extract_mgmin_per_conpla(propofol_array, not_null_indices, patient_first_row)
	
	#=====Approach 2 : summarize===
# 	all_mgmin_avg   = np.sum(all_mgmin_array, axis = 1).reshape(-1, 1)
# 	tot_mg_min      = np.array(tot_mg_min).reshape(-1, 1)
# 	all_mgmin_array = np.concatenate((all_mgmin_avg,  tot_mg_min) , axis =1)
# 	max_num_mgmin   = 2
	#==========================
	
	all_mgmin_df = pd.DataFrame(all_mgmin_array).reset_index(drop=True) # just to keep the order of the columns intact
	# =======================================================================
	# CHANGE : interpolate using tot_mg_min in  all_mgmin_array
	size         = int(chunksize)
	tolerance    = 10
	# scaler     = MinMaxScaler() 
	padBeforeIndxs = getFirstDosageIndex( pd.concat([input_data_df, pd.DataFrame(tot_mg_min) , all_mgmin_df], axis=1) , len(input_data_df.columns), size, tolerance)
	# all_mgmin_array = np.cumsum(all_mgmin_array, axis = 1)
	# all_mgmin_array = (all_mgmin_array - all_mgmin_array.min(axis = 1).reshape(-1,1) )/ (all_mgmin_array.max(axis = 1).reshape(-1,1)  - all_mgmin_array.min(axis = 1).reshape(-1,1) + 1e-10)  #Approach : normalize infusions 
	all_mgmin_array_resampled, discard_indxs = resample_data(all_mgmin_array, tot_mg_min, padBeforeIndxs, size, tolerance)
	max_num_mgmin = size
	all_mgmin_df = pd.DataFrame(all_mgmin_array_resampled).reset_index(drop=True)
	print("discard indx length : ", len(discard_indxs))

	# Perform running cummulative sum of mg_min along rows (axis=1)
    # all_mgmin_df = all_mgmin_df.cumsum(axis=1)  # performed above before interpolation
	# ========================================================================

	trainingdata = pd.concat([input_data_df, all_mgmin_df], axis=1)
	# checkPlot(pd.concat([input_data_df, pd.DataFrame(tot_mg_min) , all_mgmin_df], axis=1))
	
	#========== synthetic data append ;  learn zero concentration prediction ========================
	for pid in patient_ids:
		new_row        = np.zeros(trainingdata.shape[1])
		new_row[:7]    = np.array(trainingdata[trainingdata.iloc[:, 0] == pid].iloc[0, :7])
		for _ in range(1): # negative sampling
			new_row[1]     = trainingdata.iloc[:, 1].sample(n = 1, random_state = 1).values #WT
			new_row[2]     = trainingdata.iloc[:, 2].sample(n = 1, random_state = 1).values #HT
			new_row[3]     = trainingdata.iloc[:, 3].sample(n = 1, random_state = 1).values #AGE
			new_row[4]     = trainingdata.iloc[:, 4].sample(n = 1, random_state = 1).values #SEX	
			trainingdata.loc[len(trainingdata)] = new_row

	# =======================================================================
	# CHANGE : Discard rows in training data 
	trainingdata.drop(discard_indxs, inplace = True)
	print("total training data size : ", trainingdata.shape)
	# =======================================================================  # "CMT", "MDV", "EVID", -- 
	
	original_column_names = ["ID", "WT", "HT", "AGE", "SEX", "OPIOD", "STUDY",  "CON_PLA", "PREV_CON_PLA"]  # ["ID", "WT", "HT", "AGE", "SEX", "OPIOD", "STUDY", "CON_PLA", "PREV_CON_PLA"] 
	time_column_names = ["Time" + str(num) for num in range(max_num_mgmin)]                                 # time-0 to time-xyz
	trainingdata.columns = original_column_names + time_column_names                                        # 
	if not prepare_test:
		trainingdata.to_excel(utils.data_path + 'trainingdata.xlsx')                                        # FINAL TRAINING DATA FILE

	return np.asarray(trainingdata), list(trainingdata.columns)
	
'''
Brief   : Method to interpolate data on a equidistant time plane
Input   : input data, total length of each input data sample, target size, tolerance : above or below which to not interpolate
Returns : returns np.array of size (N,Size)
'''
def resample_data(data, tot_mg_min, padBeforeIndxs, size = 30, tolerance = 5):

	print("in resample_data")
	print(f" prev data shape : {data.shape}")

	n = data.shape[0]
	discard_indxs = []
	data_resampled = np.zeros((n, size))

	for i in range(n):

		tot_mg_min_i = tot_mg_min[i]
		if tot_mg_min_i <= size + tolerance and tot_mg_min_i >= size - tolerance:
			# interpolate when within tolerance
			new_x = np.linspace(1, tot_mg_min_i, size)
			xp = np.arange(1, tot_mg_min_i+1)
			yp = data[i, :tot_mg_min_i]
			data_resampled[i] = np.interp(new_x, xp, yp)
		elif tot_mg_min_i < size - tolerance:
			# pad when below tolerance
			data_resampled[i] = data[i, :size]  # contains cumsum for the entire length of array
			if i in padBeforeIndxs:
				#pad before with 0 ;  
				lenToPad = size - tot_mg_min_i
				data_resampled[i] = np.pad(data_resampled[i, :tot_mg_min_i], (lenToPad, 0), 'constant', constant_values = 0)
			else:
				#pad after
				data_resampled[i, tot_mg_min_i : size ] = 0 # replace with 0 
		else:
			# discard when above tolerance
			discard_indxs.append(i)

	print(f" resampled data shape : {data_resampled.shape}")
		
	return data_resampled, discard_indxs


'''
Brief   : Method to call the preprocessing method and then split the preprocessed data into train+test
Input   : Path to the raw data
Returns : Train and test input and targets, features of training data (ht, wt, age, etc)
'''
def get_data(path):
	# preprocess the data before training
	trainingdata, header = preprocess_data(path)
	
	# CON_PLA is the target    
	target = trainingdata[:, 7] 

	# delete ID, CON_PLA
	input = np.delete(trainingdata, np.s_[0,7,8], axis=1)  
	input_features = header[1:7] + header[9:] 
	print("Input shape : ", input.shape) 
	
	# Split (input,target) into train/test 
	train_input, test_input, train_target, test_target = train_test_split(input,
	                                                                      target,
	                                                                      test_size = 0.15,
	                                                                      random_state = 42)  #8

	print("Train data size : {}\nTest data size : {}".format(train_input.shape[0], test_input.shape[0]))

	return train_input, test_input, train_target, test_target, input_features

def ret_t(x, l):
	if l < 0:
		return None
	elif l >= 0 and x[l] > 0 :
		return l
	else : 
		return ret_t(x, l-1)


def get_testData(path):

	try:
		propofol_df = pd.read_csv(path, float_precision='round_trip')
	except:
		print("Test data file does not exist or is broken! Run dataloader.get_data() before running dataloader.get_testData() ")

	test_data = np.zeros((1, (int)(chunksize + 7) )) # 10 for the patient indicator features

	for pid in propofol_df.ID.unique():
		patient_data      = propofol_df[propofol_df.ID == pid]
		patient_ind       = np.array(patient_data[["ID", "kg", "cm", "years", "SEX", "Opioid ", "Study"]].iloc[0]) #, "CMT", "MDV", "EVID"
		infusions         = np.array(patient_data["mg/min"])
		# infusions 		  = np.cumsum(infusions)                 # aggregate all infusions per patient until the end of surgery
		total_chunks      = np.ceil(infusions.shape[0] / chunksize)  # get ceiling on the total rows for infusion matrix
		infusions.resize(int(total_chunks), int(chunksize))			 # resize flat infusion array to infusion matrix with column size equal to chucksize
		infusions         = np.delete(infusions, np.where(~infusions.any(axis=1))[0], axis = 0)
		# infusions         = (infusions - infusions.min(axis = 1).reshape(-1,1) ) / (infusions.max(axis = 1).reshape(-1,1)  - infusions.min(axis = 1).reshape(-1,1) )  #Approach : normalize infusions 
		zero_infusion_idx = ret_t(infusions[-1], int(chunksize)-1)   # get from backwards the first index of element which is greater than zero ; set all elements after this index to zero to avoid cumsum effect 
		# infusions         = np.cumsum(infusions, axis = 1)			 # aggregate propoful infusions by row/observed chunk
		if zero_infusion_idx is not None and zero_infusion_idx < chunksize-1:
			infusions[-1, zero_infusion_idx + 1: ] = 0
		patient_ind       = np.broadcast_to(patient_ind, (infusions.shape[0], 7) )  # broadcast patient indicator data to size of infusion matrix
		patient_data      = np.concatenate((patient_ind, infusions), axis = 1)      # concatenate patient indicator data and infusion 
		test_data         = np.concatenate((test_data, patient_data), axis = 0)     # append patient data to build test data
	
	test_data = test_data[1:, :]
	# np.savetxt("test_data.csv", test_data, delimiter = ",")

	return test_data
