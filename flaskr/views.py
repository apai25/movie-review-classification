from flask import render_template, url_for, redirect, session
from flaskr import app
from flaskr.forms import UploadForm
from tensorflow.keras.models import load_model

model = load_model('model/saved_model')

@app.route('/')
def home():
    return redirect(url_for('upload'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    prediction = ''
    form = UploadForm()
    if form.validate_on_submit():
        session['text'] = form.text_upload.data
        return redirect(url_for('output'))
    return render_template('upload.html', form=form)

@app.route('/output')
def output():
    global model
    prediction = model.predict([[session['text']]])[0][0] >- 0
    return render_template('output.html', prediction=prediction)