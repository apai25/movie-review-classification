from tokenize import String
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired

class UploadForm(FlaskForm):
    
    text_upload = StringField(label='Enter your text here: ', validators=[DataRequired()])
    submit = SubmitField('Submit')
