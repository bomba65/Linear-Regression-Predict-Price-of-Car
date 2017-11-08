import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn import model_selection
from sklearn.metrics import accuracy_score
'''
iris = load_iris()

x = iris.data
y = iris.target

gnd = GaussianNB()

gnd.fit(x,y)

y_pred = gnd.predict(x)

print(accuracy_score(y_pred,y))

data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
'''

data = pd.ExcelFile("task1.xlsx")
train = data.parse("TRAIN")
train.set_index('ID', inplace=True)

test = data.parse("TEST")
test.set_index('ID', inplace=True)

x = train.loc[:,:'AVG_COST']
y = train['ESTIM_COST']

x['AUTO_CONDITION'] = pd.Categorical(x['AUTO_CONDITION'])
#x['AUTO_CONDITION']=x['AUTO_CONDITION'].map({ 'Удовлетворительное': 0.0, 'Хорошее': 1.0, 'Отличное': 2.0})
x['TRANSM_TYPE'] = x['TRANSM_TYPE'].astype('category')
x['INTERIOR_TYPE'] = x['INTERIOR_TYPE'].astype('category')
x['TYPE_OF_DRIVE'] = x['TYPE_OF_DRIVE'].astype('category')
x['BODY_TYPE'] = x['BODY_TYPE'].astype('category')
x['FUEL_TYPE'] = x['FUEL_TYPE'].astype('category')
x['VIN_3'] = x['VIN_3'].astype('category')
x['VIN_2'] = x['VIN_2'].astype('category')
x['VIN_1'] = x['VIN_1'].astype('category')

test['TRANSM_TYPE'] = test['TRANSM_TYPE'].astype('category')
test['INTERIOR_TYPE'] = test['INTERIOR_TYPE'].astype('category')
test['TYPE_OF_DRIVE'] = test['TYPE_OF_DRIVE'].astype('category')
test['BODY_TYPE'] = test['BODY_TYPE'].astype('category')
test['FUEL_TYPE'] = test['FUEL_TYPE'].astype('category')
test['VIN_3'] = test['VIN_3'].astype('category')
test['VIN_2'] = test['VIN_2'].astype('category')
test['VIN_1'] = test['VIN_1'].astype('category')
test['AUTO_CONDITION'] = test['AUTO_CONDITION'].astype('category')

def convert_to_ascii(text):
    return "".join(str(ord(char)) for char in text)

cat_columns = x.select_dtypes(['category']).columns
x["VIN_1"] = x["VIN_1"].apply(lambda x: ord(x))
x["VIN_2"] = x["VIN_2"].apply(lambda x: ord(x))
x["VIN_3"] = x["VIN_3"].apply(lambda x: ord(x))

cat_columns = test.select_dtypes(['category']).columns
test[cat_columns] = test[cat_columns].apply(lambda test: test.cat.codes)

from sklearn.preprocessing import normalize
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#x = (x - x.mean()) / (x.max() - x.min())
#test = (test - test.mean()) / (test.max() - test.min())
#y = (y - y.mean()) / (y.max() - y.min())
x.fillna(value=0, inplace=True) 
test.fillna(value=0, inplace=True)

#print(x.info())



from sklearn.model_selection import train_test_split

lr = linear_model.LinearRegression()
gnd = GaussianNB()
logreg = linear_model.LogisticRegression(C=1e5)

from sklearn.neural_network import MLPRegressor

nn = MLPRegressor(
    hidden_layer_sizes=(100,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#lr.fit(x,y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
nn.fit(X_train, y_train)
predicted = nn.predict(X_test)
predicted2 = predicted//1000*1000
predicted2 = predicted2.astype(int)
print(accuracy_score(predicted2, y_test))

acc = 0
j = 0
for i in y_test:
    pr = i * 0.1
    if predicted2[j] >= i - pr and predicted2[j] <= i + pr:
        acc = acc + 1
    j = j + 1

print(acc/len(y_test))

'''
predicted = lr.predict(test)
predicted2 = predicted//1000*1000
predicted2 = predicted2.astype(int)

predicted3 = pd.DataFrame(data=predicted2,index=test.index)
predicted3=predicted3.rename(columns = {0:'ESTIM_COST'})
writer = pd.ExcelWriter('Answer.xlsx', engine='xlsxwriter')
predicted3.to_excel(writer, sheet_name='Test')
writer.save()
'''