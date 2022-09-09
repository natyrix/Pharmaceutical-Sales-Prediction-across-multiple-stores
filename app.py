from flask import *  
app = Flask(__name__, static_folder='staticFiles')  


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
            return render_template('prediction_result.html', title='Prediction Result')
    except Exception as e:
        return render_template('predict.html', title='Error', error_message=str(e))


@app.route('/technical_notes')
def technical_notes():
    return render_template('technical_notes.html', title='Technical Notes')




if __name__ == '__main__':  
   app.run(debug = True)  