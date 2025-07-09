from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model_202211044 (1).pkl')
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            features = [
                float(request.form['math']),
                float(request.form['reading']),
                float(request.form['writing'])
            ]
            df = pd.DataFrame([features], columns=['math score', 'reading score', 'writing score'])
            prediction = model.predict(df)[0]
        except:
            prediction = "Error: Please check your input values."
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
