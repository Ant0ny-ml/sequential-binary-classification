import pandas_profiling as pp
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
import sklearn
from keras.callbacks import EarlyStopping



data = pd.read_csv('Placement_Data_Full_Class.csv')

categorical_columns = data.columns[data.dtypes == 'object']
le = preprocessing.LabelEncoder()
for column in categorical_columns:
    data[column] = le.fit_transform(list(data[column]))

# pp.profile_report.ProfileReport(data).to_file(output_file='test.html')

data.drop(['salary'], axis=1, inplace=True)
print(len(data) - data.count())

predict = 'status'

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Dense(126, input_dim=len(data.columns)-1, activation='sigmoid'))
model.add(Dense(48, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="Adamax", metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(x_train, y_train, epochs = 200, batch_size=10, validation_data=(x_test, y_test), shuffle=True, callbacks=[early_stopping])

scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))