import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.ExcelFile("task1.xlsx")
train = data.parse("TRAIN")
train.set_index('ID', inplace=True)

test = data.parse("TEST")
test.set_index('ID', inplace=True)

x = train.loc[:,:'AVG_COST']
y = train['ESTIM_COST']

x['VIN_3'] = x['VIN_3'].astype('category')
x['VIN_2'] = x['VIN_2'].astype('category')
x['VIN_1'] = x['VIN_1'].astype('category')

test['VIN_3'] = test['VIN_3'].astype('category')
test['VIN_2'] = test['VIN_2'].astype('category')
test['VIN_1'] = test['VIN_1'].astype('category')

x["VIN_1"] = x["VIN_1"].apply(lambda x: ord(x))
x["VIN_2"] = x["VIN_2"].apply(lambda x: ord(x))
x["VIN_3"] = x["VIN_3"].apply(lambda x: ord(x))

test["VIN_1"] = test["VIN_1"].apply(lambda x: ord(x))
test["VIN_2"] = test["VIN_2"].apply(lambda x: ord(x))
test["VIN_3"] = test["VIN_3"].apply(lambda x: ord(x))

x['FUEL_TYPE']=x['FUEL_TYPE'].map({ 'Дизель': 0.0, 'Гибрид': 1.0, 'Газ': 2.0, 'Бензин-Газ': 3.0, 'Бензин': 4.0})
x['BODY_TYPE']=x['BODY_TYPE'].map({ 'Хэтчбек/Лифтбек': 0.0, 'Универсал': 1.0, 'Седан': 2.0, 'Пикап': 3.0, 'Минивэн': 4.0, 'Кроссовер': 5.0, 'Внедорожник': 6.0})
x['TYPE_OF_DRIVE']=x['TYPE_OF_DRIVE'].map({ 'Полный привод': 0.0, 'Передний привод': 1.0, 'Задний привод': 2.0})
x['INTERIOR_TYPE']=x['INTERIOR_TYPE'].map({ 'ВЕЛЮР': 0.0, 'КОЖА': 1.0, 'КОМБИНИРОВАННЫЙ': 2.0})
x['TRANSM_TYPE']=x['TRANSM_TYPE'].map({ 'МКПП': 0.0, 'АКПП': 1.0})
x['AUTO_CONDITION']=x['AUTO_CONDITION'].map({ 'Удовлетворительное': 0.0, 'Хорошее': 1.0, 'Отличное': 2.0})

test['FUEL_TYPE']=test['FUEL_TYPE'].map({ 'Дизель': 0.0, 'Гибрид': 1.0, 'Газ': 2.0, 'Бензин-Газ': 3.0, 'Бензин': 4.0})
test['BODY_TYPE']=test['BODY_TYPE'].map({ 'Хэтчбек/Лифтбек': 0.0, 'Универсал': 1.0, 'Седан': 2.0, 'Пикап': 3.0, 'Минивэн': 4.0, 'Кроссовер': 5.0, 'Внедорожник': 6.0})
test['TYPE_OF_DRIVE']=test['TYPE_OF_DRIVE'].map({ 'Полный привод': 0.0, 'Передний привод': 1.0, 'Задний привод': 2.0})
test['INTERIOR_TYPE']=test['INTERIOR_TYPE'].map({ 'ВЕЛЮР': 0.0, 'КОЖА': 1.0, 'КОМБИНИРОВАННЫЙ': 2.0})
test['TRANSM_TYPE']=test['TRANSM_TYPE'].map({ 'МКПП': 0.0, 'АКПП': 1.0})
test['AUTO_CONDITION']=test['AUTO_CONDITION'].map({ 'Удовлетворительное': 0.0, 'Хорошее': 1.0, 'Отличное': 2.0})

from sklearn.preprocessing import normalize
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

x['VIN_3'] = x['VIN_3'].astype('float')
x['VIN_2'] = x['VIN_2'].astype('float')
x['VIN_1'] = x['VIN_1'].astype('float')

test['VIN_3'] = test['VIN_3'].astype('float')
test['VIN_2'] = test['VIN_2'].astype('float')
test['VIN_1'] = test['VIN_1'].astype('float')

x.fillna(value=0, inplace=True)
test.fillna(value=0, inplace=True)

lr = linear_model.LinearRegression()
lr.fit(x, y)

predicted = lr.predict(test)
predicted2 = predicted//1000*1000
predicted2 = predicted2.astype(int)

predicted3 = pd.DataFrame(data=predicted2,index=test.index)
predicted3=predicted3.rename(columns = {0:'ESTIM_COST'})
writer = pd.ExcelWriter('Answer3.xlsx', engine='xlsxwriter')
predicted3.to_excel(writer, sheet_name='TEST')
writer.save()