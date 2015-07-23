# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:24:36 2015

@author: Jake
"""

#Script that takes the average predicitions of a number of models and averages them.

pred_df = pd.DataFrame({'gbm_log_mod': gbm_log_bag_predicted, 'rf_log_mod': rf_log_mod_predicted, 
                        'svm_log_mod': svm_log_mod_predicted,
                        'et_log_mod' : et_log_mod_predicted, 
                        'gbm_mod': gbm_bag_predicted, 
                        'rf_mod': rf_mod_predicted,
                        'svm_mod': svm_mod_predicted, 
                        'et_mod': et_mod_predicted,
                        'lasso_mod': lasso_mod_predicted,
                        'elastic_net': enet_mod_predicted})
                        
pred_df['average'] = pred_df.mean(axis = 1)        

averaged_predictions = pred_df['average']

submission = pd.DataFrame({'Id': test_id, 'Hazard': averaged_predictions})
submission = submission.set_index('Id')
submission.to_csv('averaged.csv')


                