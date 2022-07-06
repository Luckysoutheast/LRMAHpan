from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField,BooleanField
from wtforms.widgets import TextArea
from flask_wtf.file import FileField


class QueryForm(FlaskForm):
    peptide = StringField('peptide', widget=TextArea())
    mhc = StringField('MHC')
    neoantigen = BooleanField()
    save = BooleanField()
    ma = BooleanField()
    file_upload = FileField()
    submit_button = SubmitField('Submit')




