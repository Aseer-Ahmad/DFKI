"""
@purpose : Main file to initiate all the tasks and train the model. Please run this file in order to compile and excute the code
@when : 09/01/22
NOTE : Refer to the description of utils.py before running this file
"""

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

# import local files here
import dataloader
from randomforest import RandomForest
from neuralnet import trainNN, evalNN
from svm_custom import SVM_C
from gradientBoosting import GradientBoost_C
import utils
import pickle

import warnings
warnings.filterwarnings("ignore")
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

# Fetch hyper-params from utils.py
args = utils.parser.parse_args()

# Fetch train and test data from dataloader
train_input, test_input, train_target, test_target, input_features = dataloader.get_data(args.data_path + utils.raw_data)
testdata_bystudy = dataloader.get_testData("test_data_by_study.csv")
testdata_bystudy_raw = pd.read_csv("test_data_by_study.csv", float_precision='round_trip')

print('Data loaded and preprocessed')

# Choose the model as per user input - Randomforest / NN ?
print("\nRunning " + args.model)
if args.model == "rf":
    
    model_obj = RandomForest()
    result = pd.DataFrame(columns = ["n_estimators", "max_depth", "mdpe", "mdape", "r2"])
    best_model = None
    best_r2  = .6
    comb_met = np.inf
    i = 0

    for number_of_trees in [16, 24, 32, 64, 128 ]:

        for max_depth in [3, 6, 9, 15, 21, 24, 27]:

            model_obj = RandomForest()
            model_obj.set_parameters(number_of_trees,max_depth)
            mdpe, mdape, r2, model = model_obj.run(train_input, test_input, train_target, test_target, input_features)
            result.loc[i] = [number_of_trees, max_depth, mdpe, mdape, r2 ]
            i += 1
            comb = np.abs(mdpe + mdape)
            if comb < comb_met and r2 > best_r2:
                # best_r2    = r2
                comb_met   =  comb
                print(f"best model with mdpe : {mdpe} mdape : {mdape} and r2 : {r2}")
                best_model = model

    print("\nBest model output on test data")
    print(model_obj.eval(best_model, test_input, test_target, show_out = True, fname = "test"))

    print("\nBest model output on train data")
    print(model_obj.eval(best_model, train_input, train_target, show_out = True, fname = "train"))

    print("RESULT FROM GRID SEARCH \n ", end = '\n')
    print(result.sort_values('r2', ascending = False) )
    
    # filename = 'rf_best_model_T25_10.sav'
    # pickle.dump(best_model, open(filename, 'wb'))
    # print("\n best model saved")

    model_obj.predictConPlaVector(best_model, testdata_bystudy, testdata_bystudy_raw)



elif args.model == "nn":
    print("Initialize NN model")
    
    # initialize hyperparameters
    hyperparameters = {
        'num_epochs' 			: 15,
        'batch_size' 			: 32,
        'learning_rate'		    : 9e-4,
        'learning_rate_decay'   : 0.95,
        'reg'					: 0.001,
        'training_per' 		    : 0.8, 
        'input_size'            : len(input_features), 
        'hidden_size'           : [20, 10], 
        'out'                   : 1
    }

    model_obj, device = trainNN(train_input, train_target, hyperparameters, val = True)
    evalNN(model_obj, test_input, test_target, device)

elif args.model == "svm":
    model_obj = SVM_C()
    result = pd.DataFrame(columns = ["kernel", "degree", "C", "epsilon", "mdpe", "mdape", "r2"])
    best_model = None
    best_r2 = 0
    i = 0

    for kernel in ["linear", "poly", "rbf", "sigmoid"]:
        for degree in [ 2, 3,]:
                for C in [1, 2]:
                    for epsilon in [0.1, 0.2]:

                        model_obj = SVM_C()
                        model_obj.set_parameters(kernel, degree,  C , epsilon )
                        mdpe, mdape, r2 = model_obj.run(train_input, test_input, train_target, test_target, input_features)
                        result.loc[i] = [kernel, str(degree), str(C) , str(epsilon), str(mdpe), str(mdape), str(r2) ]
                        i += 1
                        if r2 > best_r2:
                            best_r2 = r2
                            best_model = model_obj

    print("RESULT FROM GRID SEARCH \n ", end = '\n')
    print(result.sort_values('r2', ascending = False) )

elif args.model == "gb":
    
    parameters = {
        'min_samples_leaf':range(30,71,10),
        'max_depth':range(5,16,2),
        'min_samples_split':range(200,2100,200),
        'n_estimators' : range(10,81,10),
        'max_features' : ["auto"],
        'subsample': [0.6,0.7,0.75,0.8,0.85,0.9]        
        }

    model = GradientBoost_C()
    model.set_parameters(parameters)
    mdpe, mdape, r2 , grid_scores = model.run(train_input, test_input, train_target, test_target, input_features)
    grid = pd.DataFrame(grid_scores)

    print(mdpe, mdape, r2)
    print(grid.loc[:, ["params", "std_test_score", "rank_test_score", "mean_test_score" ] ].sort_values(by='rank_test_score', ascending=True) )
else:
    print("Model choice unavailable.")

   
        

# importances = best_model.rf.feature_importances_
# indices = np.argsort(importances)
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [input_features[i] for i in indices])
# plt.xlabel('Relative Importance - old data transformed')
# plt.savefig("test.jpg")
# plt.show()


