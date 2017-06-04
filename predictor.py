import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error
from lstm import generate_data, lstm_model, load_csvdata
import datetime
import matplotlib.dates as mdates

LOG_DIR = '/home/tromesh/MyWork/Sedule_Predictor/model'
TIMESTEPS = 10
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

def load_attendence_frame(filename):
    #load the attendece data and make a date
    data_raw = pd.read_csv(filename, dtype={'Time': str, 'Date': str})
    data_raw['Attendance'] = data_raw['Attendance'].astype(float)
    times = []
    for index, row in data_raw.iterrows():
        _t = datetime.time(int(row['Time'][:2]), int(row['Time'][:-2]), 0) #2153
        _d = datetime.datetime.strptime( row['Date'], "%Y%m%d" ) #20150905
        times.append(datetime.datetime.combine(_d, _t))

    data_raw['_time'] = pd.Series(times, index=data_raw.index)
    df =  pd.DataFrame(data_raw, columns=['_time','Attendance'])
    return df.set_index('_time')


# scale values to reasonable values and convert to float
data_attendence = load_attendence_frame("/home/tromesh/MyWork/Sedule_Predictor/data/lecturedata.csv")
X, y = load_csvdata(data_attendence, TIMESTEPS, seperate=False)


# create a lstm instance and validation monitor
regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
                           model_dir=LOG_DIR)


# create a lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                     every_n_steps=PRINT_STEPS,
                                                     early_stopping_rounds=1000)
regressor.fit(X['train'], y['train'],
              monitors=[validation_monitor],
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)

predicted = regressor.predict(X['test'])


#cast predicted to array
predictedArr = np.array(list(predicted))
predictedArr.shape = -1,1

#not used in this example but used for seeing deviations
rmse = np.sqrt(((predictedArr - y['test']) ** 2).mean(axis=0))

score = mean_squared_error(predictedArr, y['test'])
print ("MSE: %f" % score)




# plot the data
all_dates = data_attendence.index.get_values()

fig, ax = plt.subplots(1)
fig.autofmt_xdate()

predicted_values = predictedArr.flatten() #already subset
predicted_dates = all_dates[len(all_dates)-len(predicted_values):len(all_dates)]
predicted_series = pd.Series(predicted_values, index=predicted_dates)
plot_predicted, = ax.plot(predicted_series, label='predicted (c)')

test_values = y['test'].flatten()
test_dates = all_dates[len(all_dates)-len(test_values):len(all_dates)]
test_series = pd.Series(test_values, index=test_dates)
plot_test, = ax.plot(test_series, label='2016 (c)')

xfmt = mdates.DateFormatter('%b %d %H')
ax.xaxis.set_major_formatter(xfmt)

ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H')
plt.title('Attendence Predictions for AI 2017 vs 2016')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
