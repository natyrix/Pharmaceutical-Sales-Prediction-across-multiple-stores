from flask import *
import pandas as pd
import numpy as np
import pickle
import os,sys
from datetime import date, datetime, timedelta
from werkzeug.utils import secure_filename
# from tensorflow.keras.models import load_model

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app = Flask(__name__, static_folder='staticFiles')  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class Predicter():
    def __init__(self) -> None:
        self.curr_date = datetime.now()
    def getDate(self):
        self.curr_date+=timedelta(days=1)
        return str(self.curr_date).split()[0]
    def preprocess(self, values, predict_6_weeks=False, from_file=False, file_path=None):
        if not from_file:
            if predict_6_weeks:
                data = {
                    "StateHoliday": [0 for i in range(42)],
                    "Store": [values['store_id'] for i in range(42)],
                    "Open": [1 for i in range(42)],
                    "Promo": [values['is_promo'] for i in range(42)],
                    "Date": [self.getDate() for i in range(42)],
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
        else:
            try:
                df = pd.read_csv(file_path, parse_dates=True, index_col="Date")
                df['Year'] = df.index.year
                df['Month'] = df.index.month
                df['Day'] = df.index.day
                df['WeekOfYear'] = df.index.weekofyear
                return df
            except Exception as e:
                raise Exception(str(e))
        # data = {
        #     "StateHoliday": [np.asarray(values['is_holiday'])],
        #     "Store": [np.asarray(values['store_id'])],
        #     "Open": [np.asarray(1)],
        #     "Promo": [np.asarray(values['is_promo'])],
        #     "Date": [values['date']],
        #     "SchoolHoliday": [np.asarray(values['is_school_holiday'])],
        # }
        df = pd.DataFrame(data)
        fd = df['Date']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['WeekOfYear'] = df.index.weekofyear
        my_d = [date.fromisoformat(str(i).split()[0]) for i in fd]
        df['DayOfWeek'] = [d.isoweekday() for d in my_d]
        df['weekday'] = [1 if d.isoweekday() not in [6,7] else 0 for d in my_d]
        return df
    def predict_random_forest(self, df):
        loaded_model = None
        customer_model = None
        sales_model = None
        # with open("./models/random_forest2022-09-10-01:20:33.pkl", 'rb') as f:
        #     loaded_model = pickle.load(f)
        with open("./models/random_forest2022-09-10-02:43:12.pkl.pkl", 'rb') as f:
            loaded_model = pickle.load(f)
        # with open("./models/random_forest_sales_2022-09-11-00:25:56.pkl", 'rb') as f:
        #     sales_model = pickle.load(f)
        # with open("./models/random_forest_customers_2022-09-11-00:28:36.pkl", 'rb') as f:
        #     customer_model = pickle.load(f)
        print(loaded_model)
        df = df[['StateHoliday', 'Store', 'DayOfWeek', 'Open', 'Promo',
                'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekOfYear']]
        result = loaded_model.predict(df)
        # customer_result = customer_model.predict(df)
        result = np.exp(result)
        result_dict = {
            'Date': df.index.values,
            'Store':df.Store,
            'Predicted Sales': result,
            # 'Customer': customer_result
        }
        result_df = pd.DataFrame(result_dict)
        ss = str(datetime.now()).split()[1]
        result_df.to_csv(f'staticFiles/result{ss}.csv')
        print("SAVED")
        # result_dict = {
        #     'Date': df.index.values.tolist(),
        #     'Predicted Sales': result.tolist()
        # }
        returned_result = []
        for r in [result_dict]:
            for d, s, st in zip(r['Date'], r['Predicted Sales'], r['Store']):
                returned_result.append(
                    {
                        'Date': str(d).split('T')[0],
                        'Store':st,
                        'Predicted Sales': s,
                        # 'Customer': cs
                    }
                )
        return returned_result , ss
    # def predict_tensorflow(self, df):
    #     model = load_model('./models/LSTM_sales 2022-09-09-17:05:36.pkl')
    #     df = df[['StateHoliday', 'Store', 'DayOfWeek', 'Open', 'Promo',
    #             'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekOfYear','weekday']]
    #     df = np.asarray(df).astype(np.float32)
    #     result = model.predict(df)
    #     print(np.exp(result))


@app.route('/')
@app.route('/eda')
def eda():
    return render_template('EDA.html', title='EDA')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # try:
        if request.method == "GET":
            return render_template('predict.html', title='Predict')
        elif request.method == "POST":
            store_id = request.form['store_id']
            dt = request.form.get('date')
            week6=False
            if not dt:
                week6=True
            predictor = Predicter()
            
            df = predictor.preprocess(request.form, predict_6_weeks=week6)
            # predictor.predict_tensorflow(df)
            result, fn = predictor.predict_random_forest(df)
            return render_template('predict.html', title='Prediction Result', result=result, filen=fn)
    # except Exception as e:
    #     return render_template('error_page.html', title='Error', error_message=str(e))


@app.route('/upload_and_test', methods=['POST'])
def upload_and_test():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                raise Exception("No, file part")
            file = request.files['file']
            if file.filename == '':
                raise Exception('No selected file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                full_path = os.path.join(
                        app.config['UPLOAD_FOLDER'], filename)
                predictor = Predicter()
                df = predictor.preprocess({}, from_file=True, file_path=full_path)
                # predictor.predict_tensorflow(df)
                result, fn = predictor.predict_random_forest(df)
                return render_template('predict.html', title='Prediction Result', result=result, filen=fn)
    except Exception as e:
        return render_template('error_page.html', title='Error', error_message=str(e))




if __name__ == '__main__':  
   app.run(debug = True)  