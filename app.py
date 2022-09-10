from flask import *
import pandas as pd
import numpy as np
import pickle
from datetime import date
# from tensorflow.keras.models import load_model
app = Flask(__name__, static_folder='staticFiles')  


class Predicter():
    def preprocess(self, values, predict_6_weeks=False):
        if predict_6_weeks:
            data = {
                "StateHoliday": [0 for i in range(42)],
                "Store": [values['store_id'] for i in range(42)],
                "Open": [1 for i in range(42)],
                "Promo": [values['is_promo'] for i in range(42)],
                "Date": [values['date']],
                "SchoolHoliday": [0 for i in range(42)],
            }
        else:
            data = {
                "StateHoliday": [values['is_holiday']],
                "Store": [values['store_id']],
                "Open": [1],
                "Promo": [values['is_promo']],
                "Date": [values['date']],
                "SchoolHoliday": [values['is_school_holiday']],
            }
        # data = {
        #     "StateHoliday": [np.asarray(values['is_holiday'])],
        #     "Store": [np.asarray(values['store_id'])],
        #     "Open": [np.asarray(1)],
        #     "Promo": [np.asarray(values['is_promo'])],
        #     "Date": [values['date']],
        #     "SchoolHoliday": [np.asarray(values['is_school_holiday'])],
        # }
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['WeekOfYear'] = df.index.weekofyear
        my_d = date.fromisoformat(values['date'])
        df['DayOfWeek'] = my_d.isoweekday()
        df['weekday'] = 1 if my_d.isoweekday() not in [6,7] else 0
        return df
    def predict_random_forest(self, df):
        loaded_model = None
        # with open("./models/random_forest2022-09-10-01:20:33.pkl", 'rb') as f:
        #     loaded_model = pickle.load(f)
        with open("./models/random_forest2022-09-10-02:43:12.pkl.pkl", 'rb') as f:
            loaded_model = pickle.load(f)
        print(loaded_model)
        df = df[['StateHoliday', 'Store', 'DayOfWeek', 'Open', 'Promo',
                'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekOfYear']]
        result = loaded_model.predict(df)
        result = np.exp(result)
        result_dict = {
            'Date': df.index.values,
            'Predicted Sales': result
        }
        result_df = pd.DataFrame(result_dict)
        result_df.to_csv('staticFiles/result.csv')
        print("SAVED")
        # result_dict = {
        #     'Date': df.index.values.tolist(),
        #     'Predicted Sales': result.tolist()
        # }
        returned_result = []
        for r in [result_dict]:
            for d, s in zip(r['Date'], r['Predicted Sales']):
                returned_result.append(
                    {
                        'Date': str(d).split('T')[0],
                        'Predicted Sales': s
                    }
                )
        return returned_result
    # def predict_tensorflow(self, df):
    #     model = load_model('./models/LSTM_sales 2022-09-09-17:05:36.pkl')
    #     df = df[['StateHoliday', 'Store', 'DayOfWeek', 'Open', 'Promo',
    #             'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekOfYear','weekday']]
    #     df = np.asarray(df).astype(np.float32)
    #     result = model.predict(df)
    #     print(np.exp(result))




@app.route('/')
@app.route('/index')
def index():
    name = 'Rosalia'
    return render_template('index.html', title='Sales Prediction', username=name)


@app.route('/eda')
def eda():
    return render_template('EDA.html', title='EDA')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == "GET":
            return render_template('predict.html', title='Predict')
        elif request.method == "POST":
            store_id = request.form['store_id']
            date = request.form['date']
            print("########")
            print(store_id)
            print(date)
            predictor = Predicter()
            df = predictor.preprocess(request.form)
            # predictor.predict_tensorflow(df)
            result = predictor.predict_random_forest(df)
            return render_template('predict.html', title='Prediction Result', result=result)
    except Exception as e:
        return render_template('error_page.html', title='Error', error_message=str(e))


@app.route('/technical_notes')
def technical_notes():
    return render_template('technical_notes.html', title='Technical Notes')




if __name__ == '__main__':  
   app.run(debug = True)  