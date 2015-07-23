# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:05:02 2015

@author: Jake
"""

#xgboost benchmark

train = pd.read_csv('train.csv', index_col = 0)
test = pd.read_csv('test.csv', index_col = 0)
test_id = test.index

X_train = train.drop(['Hazard'], axis = 1)
Y_train = train['Hazard']

X_train = np.array(X_train)
test = np.array(test)


for i in range(X_train.shape[1]):
    if type(X_train[1,i]) is str:
        lbl = LabelEncoder()
        lbl.fit(list(X_train[:,i]))
        X_train[:,i] = lbl.transform(X_train[:,i])

test = np.array(test)
for i in range(test.shape[1]):
    if type(test[1,i]) is str:
        lbl = LabelEncoder()
        lbl.fit(list(test[:,i]))
        test[:,i] = lbl.transform(test[:,i])

params = {"objective": "reg:linear",          
          "max_depth": 5,
          "seed": 1}

est = xgb.train(params, xgb.DMatrix(X_train, np.log(Y_train)))

predictions = np.exp(est.predict(xgb.DMatrix(test)))



submission = pd.DataFrame({'Id': test_id, 'Hazard': predictions})
submission = submission.set_index('Id')
submission.to_csv('xgboost.csv')