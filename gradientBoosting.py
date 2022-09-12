"""
@purpose : Class for Gradient Boosting model to initiate, train and evaluate
@when : 14/7/22
"""
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

class GradientBoost_C(object):
    
    def __init__(self):
        self.gb = GradientBoostingRegressor(random_state=2)
        
    def set_parameters(self, parameters):
        self.parameters = parameters 
		
	# Training phase
    def train(self, train_input, train_target, input_features):
        clf = GridSearchCV(self.gb, self.parameters, scoring='r2', cv=3) # neg_mean_squared_error
        clf.fit(train_input, train_target)
        self.gb = clf.best_estimator_
        self.grid_scores = clf.cv_results_
        return self.gb

    # Eval phase on validation data - compute metrics (mdpe, mdape, r2) on this
    def eval(self, model, test_input, test_target):
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

        return mdpe, mdape, r2 , self.grid_scores
        


