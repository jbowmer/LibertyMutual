# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:34:39 2015

@author: Jake
"""

#Ensemble:

X_blend = pd.DataFrame(X_blend)
X_test = pd.DataFrame(X_test)

gbm_log_bag_predictions = np.exp(gbm_log_bag.predict(X_blend)) 
rf_log_mod_predictions =np.exp(rf_log_mod.predict(X_blend))  
svm_log_mod_predictions = np.exp(svm_log_mod.predict(X_blend))  
et_log_mod_predictions = np.exp(et_log_mod.predict(X_blend))  
enet_mod_predictions = np.exp(enet_mod.predict(X_blend))  
lasso_mod_predictions = np.exp(lasso_mod.predict(X_blend))  
rf_mod_predictions = rf_mod.predict(X_blend) 
et_mod_predictions = et_mod.predict(X_blend)
svm_mod_predictions = svm_mod.predict(X_blend) 
gbm_bag_predictions = gbm_bag.predict(X_blend)  
 

#Append to blend dataset
X_blend['gbm_log_bag'] = gbm_log_bag_predictions
X_blend['rf_log_mod'] = rf_log_mod_predictions
X_blend['svm_log_mod'] =svm_log_mod_predictions
X_blend['et_log_mod'] = et_log_mod_predictions
X_blend['enet_mod'] = enet_mod_predictions
X_blend['lasso_mod'] = lasso_mod_predictions
X_blend['rf_mod'] = rf_mod_predictions
X_blend['et_mod'] = et_mod_predictions
X_blend['svm_mod'] = svm_mod_predictions
X_blend['gbm_bag'] = gbm_bag_predictions

#Test set
gbm_log_bag_predictions = np.exp(gbm_log_bag.predict(X_test)) 
rf_log_mod_predictions =np.exp(rf_log_mod.predict(X_test))  
svm_log_mod_predictions = np.exp(svm_log_mod.predict(X_test))  
et_log_mod_predictions = np.exp(et_log_mod.predict(X_test))  
enet_mod_predictions = np.exp(enet_mod.predict(X_test))  
lasso_mod_predictions = np.exp(lasso_mod.predict(X_test))  
rf_mod_predictions = rf_mod.predict(X_test) 
et_mod_predictions = et_mod.predict(X_test)
svm_mod_predictions = svm_mod.predict(X_test) 
gbm_bag_predictions = gbm_bag.predict(X_test)  
 

#Append to test dataset
X_test['gbm_log_bag'] = gbm_log_bag_predictions
X_test['rf_log_mod'] = rf_log_mod_predictions
X_test['svm_log_mod'] =svm_log_mod_predictions
X_test['et_log_mod'] = et_log_mod_predictions
X_test['enet_mod'] = enet_mod_predictions
X_test['lasso_mod'] = lasso_mod_predictions
X_test['rf_mod'] = rf_mod_predictions
X_test['et_mod'] = et_mod_predictions
X_test['svm_mod'] = svm_mod_predictions
X_test['gbm_bag'] = gbm_bag_predictions


#Train ensemble model on enhanced blend data:

ensemble_mod = GradientBoostingRegressor()
ensemble_mod.fit(X_blend, np.log(Y_blend))

#Predict on X_test
predictions = np.exp(ensemble_mod.predict(X_test))
#score
Gini(Y_test, predictions) #36.09
Gini(Y_test, gbm_log_bag_predictions) #36.21

#Submission (use test dataset):

gbm_log_bag_predictions = np.exp(gbm_log_bag.predict(test)) 
rf_log_mod_predictions =np.exp(rf_log_mod.predict(test))  
svm_log_mod_predictions = np.exp(svm_log_mod.predict(test))  
et_log_mod_predictions = np.exp(et_log_mod.predict(test))  
enet_mod_predictions = np.exp(enet_mod.predict(test))  
lasso_mod_predictions = np.exp(lasso_mod.predict(test))  
rf_mod_predictions = rf_mod.predict(test) 
et_mod_predictions = et_mod.predict(test)
svm_mod_predictions = svm_mod.predict(test) 
gbm_bag_predictions = gbm_bag.predict(test)  
 
test = pd.DataFrame(test)
#Append to test dataset
test['gbm_log_bag'] = gbm_log_bag_predictions
test['rf_log_mod'] = rf_log_mod_predictions
test['svm_log_mod'] =svm_log_mod_predictions
test['et_log_mod'] = et_log_mod_predictions
test['enet_mod'] = enet_mod_predictions
test['lasso_mod'] = lasso_mod_predictions
test['rf_mod'] = rf_mod_predictions
test['et_mod'] = et_mod_predictions
test['svm_mod'] = svm_mod_predictions
test['gbm_bag'] = gbm_bag_predictions

ensemble_predictions = np.exp(ensemble_mod.predict(test))

submission = pd.DataFrame({'Id': test_id, 'Hazard': ensemble_predictions})
submission = submission.set_index('Id')
submission.to_csv('ensembled.csv')