#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('employee_attrition_1.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('home.html')
#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2) 
    return render_template('home.html', prediction_text='Predicted status of the employee is :{}'.format(output))
    if prediction_text == 1.0:
        print("This employee is happy and most likely to stay with us!")
    elif prediction_text == 0.0:
        print("Uhhh ohh! This employee is showing signs of leaving!")
    else:
        print("Invalid inputs")
if __name__ == "__main__":
    app.run(debug=True)