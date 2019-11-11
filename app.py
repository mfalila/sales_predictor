#Importing all useful libraries
import numpy as np
from flask import Flask, request, render_template
import pickle

#Creating the flask app and reading our pkl file
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

#Creating our homepage
#By default '/' will render a template called index.html, which is the template containing the code to build our homepage
@app.route('/')
def home():
    return render_template('index.html')

#This POST method lets us provide some features to our pkl file so that our model will take those inputs and give us outputs
# The function uses the request library to take all inputs from the text fields of our page and stores them on the 'int_features' variable
# This function then takes input features and converts them to an array which is stored in the 'final_features' variable
# Finaly the features are used for prediction and stored in the 'prediction' variable
@app.route('/predict',methods = ['POST'])
def predict():
    '''
    This function renders results on our HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
#We then get the output and render HTML of our prediction text whilch will be replaced through our index.html file
    output  = round(prediction[0],2)
    
    return render_template('index.html', prediction_text = 'Year five sales should be ${}'.format(output))

#The main function runs the whole flask
if __name__ == "__main__":
    app.run(debug=True)
