from flask import Flask, render_template
# Flask - WTF!!!
from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, NumberRange
# Load My ML Model
from pandas import DataFrame
from joblib import load

# Initialize APP
app = Flask(__name__)
app.config["SECRET_KEY"] = 'sfje ksdf wrfj cdeo'

# The Wed Pages
@app.route('/')
def index():
    return render_template('index.html')

# Fast EDA
@app.route('/eda')
def eda():
    return render_template('eda.html')

# WTF Flask Class
class FormItem(FlaskForm):
    Age          = IntegerField(
        'Age In Years',
        validators=[DataRequired()]
        )

    Gender       = SelectField(
        'Gender', 
        choices   = [
            ("M", "Male"),
            ("F", "Female")
            ],
        validators = [DataRequired()]
        )

    DayOfTheWeek = SelectField(
        'Choose Regesteration Day',
        choices     =[
            ("Monday"   , "Monday"),
            ("Tuesday"  , "Tuesday"),
            ("Wednesday", "Wednesday"),
            ("Thursday" , "Thursday"),
            ("Friday"   , "Friday"),
            ("Saturday" , "Saturday"),
            ("Sunday"   , "Sunday")
            ], 
        validators=[DataRequired()]
        )

    Diabetes     = SelectField(
        'Is The Patient Diabetes',
        choices    = [
            (0, '0'),
            (1, '1'),
        ],
        validators = [DataRequired()]
        )

    Alcoholism   = SelectField(
        'Is The Patient Alcoholic',
        choices    = [
            (0, '0'),
            (1, '1'),
        ],
        validators=[DataRequired()]
        )

    HyperTension = SelectField(
        'Is The Patient suffering from Hypertension',
        choices    = [
            (0, '0'),
            (1, '1'),
        ],
        validators=[DataRequired()]
        )

    Handicap     = SelectField(
        'Is The Patient Handicapped',
        choices    = [
            (0, '0'),
            (1, '1'),
            (2, '2'),
            (3, '3'),
            (4, '4'),
        ],
        validators=[DataRequired()]
        )

    Smokes       = SelectField(
        'Does the Patient Smoke',
        choices    = [
            (0, '0'),
            (1, '1'),
        ],
        validators=[DataRequired()]
        )

    Scholarship  = SelectField(
        'If The PatientHave a Scholarship',
        choices    = [
            (0, '0'),
            (1, '1'),
        ],
        validators=[DataRequired()]
        )

    Tuberculosis = SelectField(
        'Is The Patient Suffering From Tubercolosis',
        choices    = [
            (0, '0'),
            (1, '1'),
        ],
        validators=[DataRequired()]
        )

    Sms_Reminder = SelectField(
        'Was A SMS Reminder Sent To The Student',
        choices    = [
            (0, '0'),
            (1, '1'),
            (2, '2'),
        ],
        validators=[DataRequired()]
        )

    AwaitingTime = IntegerField(
        'Awaing Time For Appointment in Days',
        validators=[DataRequired(), NumberRange(min=1, max=24, message="Number Range")]
        )

    HourOfTheDay = IntegerField(
        'Hour Of The Day',
        validators = [DataRequired(NumberRange(min=0, max=1000, message="Number Range"))]
        )

    Submit       = SubmitField('Submit')

# Prediction Form
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = FormItem()
    y_pred = None

    if form.validate_on_submit():
        
        x01 = float(form.Age.data)
        x02 = str(form.Gender.data)
        x03 = str(form.DayOfTheWeek.data)
        x04 = float(form.Diabetes.data)
        x05 = float(form.Alcoholism.data)
        x06 = float(form.HyperTension.data)
        x07 = float(form.Handicap.data)
        x08 = float(form.Smokes.data)
        x09 = float(form.Scholarship.data)
        x10 = float(form.Tuberculosis.data),
        x11 = float(form.Sms_Reminder.data)
        x12 = (-1 * float(form.AwaitingTime.data))
        x13 = float(form.HourOfTheDay.data)
        
        Form_DICT = {
            'Age'         : x01,
            'Gender'      : x02,
            'DayOfTheWeek': x03,
            'Diabetes'    : x04,
            'Alcoholism'  : x05,
            'HyperTension': x06,
            'Handicap'    : x07,
            'Smokes'      : x08,
            'Scholarship' : x09,
            'Tuberculosis': x10,
            'Sms_Reminder': x11,
            'AwaitingTime': x12,
            'HourOfTheDay': x13
        }
    
        IN_PUT = DataFrame(Form_DICT, index=[0])
        ML_Model = load(r'Model\fModel.joblib')
        y_pred = ML_Model.predict(IN_PUT)[-1]
        

    return render_template('predict.html', form = form, y_pred=y_pred)

# 404 Error Handling
@app.errorhandler(404)
def err_404(_):
    return  render_template('err_404.html')
