from cgitb import reset
from flask import *
import pandas as pd
import numpy as np
import pickle
from datetime import date
app = Flask(__name__, static_folder='staticFiles')  


class Predicter():
    def preprocess(self, values):
        data = {
            "StateHoliday": [values['is_holiday']],
            "Store": [values['store_id']],
            "Open": [1],
            "Promo": [values['is_promo']],
            "Date": [values['date']],
            "SchoolHoliday": [values['is_school_holiday']],
        }
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['WeekOfYear'] = df.index.weekofyear
        my_d = date.fromisoformat(values['date'])
        df['DayOfWeek'] = my_d.isoweekday()
        return df
    def predict(self, df):
        loaded_model = None
        with open("./models/random_forest2022-09-09-22:04:41.pkl", 'rb') as f:
            loaded_model = pickle.load(f)
        df = df[['StateHoliday', 'Store', 'DayOfWeek', 'Open', 'Promo',
                'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekOfYear']]
        result = loaded_model.predict(df)
        result = np.exp(result)
        result_dict = {
            'Date': df.index.values,
            'Predicted Sales': result
        }
        result_df = pd.DataFrame(result_dict)
        result_df.to_csv('../staticFiles/result.csv')
        return result_dict




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
            result = predictor.predict(df)
            return render_template('prediction.html', title='Prediction Result', result=result)
    except Exception as e:
        return render_template('error_page.html', title='Error', error_message=str(e))


@app.route('/technical_notes')
def technical_notes():
    return render_template('technical_notes.html', title='Technical Notes')




if __name__ == '__main__':  
   app.run(debug = True)  