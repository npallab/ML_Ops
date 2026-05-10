import sys
import joblib
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from config.path_config import MODEL_OUTPUT_PATH
from src.customexception import CustomException

app = Flask(__name__)
loaded_model = joblib.load(MODEL_OUTPUT_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            form_data = request.form
            input_data = {
                'lead_time': int(form_data['lead_time']),
                'no_of_special_requests': int(form_data['no_of_special_requests']),
                'average_price_per_room': float(form_data['average_price_per_room']),
                'arrival_month': int(form_data['arrival_month']),
                'arrival_date': int(form_data['arrival_date']),
                'market_segment_type': int(form_data['market_segment_type']),
                'no_of_week_nights': int(form_data['no_of_week_nights']),
                'no_of_weekend_nights': int(form_data['no_of_weekend_nights']),
                'type_of_meal_plan': int(form_data['type_of_meal_plan']),
                'room_type_reserved': int(form_data['room_type_reserved']),
            }

            input_df = pd.DataFrame([input_data])
            prediction = loaded_model.predict(input_df)[0]
            result = "Booking Canceled" if prediction == 1 else "Booking Not Canceled"

            return render_template('index.html', result=result)
        except Exception as e:
            raise CustomException(e, sys)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)