from flask_wtf import FlaskForm
from wtforms import SubmitField


class Start(FlaskForm):
    submit = SubmitField('Start')
