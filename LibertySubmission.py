# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 22:56:38 2015

@author: Jake
"""

#Liberty Submission script.
#Relies on previosuly trained models:


os.chdir('/Users/Jake/Kaggle/LibertyMutual/Data')

test = pd.read_csv('test.csv', index_col = 0)

test_id = test.index

X = test

X = np.array(X)

# Encode categorical variables.
for i in range(X.shape[1]):
    if type(X[1,i]) is str:
        lbl = LabelEncoder()
        lbl.fit(list(X[:,i]))
        X[:,i] = lbl.transform(X[:,i])
        
X = std_scale.transform(X)


predictions = rf_bag.predict(X)

submission = pd.DataFrame({'Id': test_id, 'Hazard': predictions})
submission = submission.set_index('Id')

submission.to_csv('meta_sub.csv')