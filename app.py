from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import traceback
import os

app = Flask(__name__)

def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at: {model_path}")
            return None
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

model = load_model()


def preprocessing(user : dict):
    user_df = pd.DataFrame([user])
    
    user_df['SessionLengthMin'] = pd.to_numeric(user_df['SessionLengthMin'])
    user_df['TotalPrompts'] = pd.to_numeric(user_df['TotalPrompts'])
    user_df['AI_AssistanceLevel'] = pd.to_numeric(user_df['AI_AssistanceLevel'])
    user_df['SatisfactionRating'] = pd.to_numeric(user_df['SatisfactionRating'])
    
    StudentLevel = {
    'Graduate' : 0,
    'High School' : 1,
    'Undergraduate' : 2
    }
    
    Discipline = {
        'Biology' : 0,
        'Business' : 1, 
        'Computer Science' : 2, 
        'Engineering' : 3, 
        'History' : 4, 
        'Math': 5,
        'Psychology':6,
        'Other':7
    }
    
    TaskType = {
        'Brainstorming' : 0, 
        'Coding' : 1, 
        'Homework Help' : 2, 
        'Research' : 3, 
        'Studying' : 4, 
        'Writing' : 5
    }
    
    FinalOutcome = {
        'Assignment Completed' : 0, 
        'Confused' : 1, 
        'Gave Up' : 2, 
        'Idea Drafted' : 3
    }
    UsedAgain = {
        'False' : 0,
        'True' : 1
    }
    
    mapping_dicts = {
        'StudentLevel': StudentLevel,
        'Discipline': Discipline,
        'TaskType': TaskType,
        'FinalOutcome': FinalOutcome,
        'UsedAgain': UsedAgain
    }
    
    for col in ['StudentLevel', 'Discipline', 'TaskType', 'FinalOutcome', 'UsedAgain']:
        user_df[col] = user_df[col].map(mapping_dicts[col])
    user_df['SatisfactionRating'] = user_df['SatisfactionRating'].apply(binRating)
    print("Processed DataFrame:")
    print(user_df)
    print("Any NaN values?")
    print(user_df.isna().sum())
    return user_df


def binRating(rating) -> int:
    if int(rating) < 0.5:
        return 0
    elif int(rating) < 1.5:
        return 1
    elif int(rating) < 2.5:
        return 2
    elif int(rating) < 3.5:
        return 3
    elif int(rating) < 4.5:
        return 4
    else: return 5
    
def predict_cluster(data: dict) -> int:
    if model is None:
        raise Exception("Model not loaded properly")
    
    print("🔄 Starting prediction process...")
    data_df = preprocessing(data)
    print(f"📊 Preprocessed data shape: {data_df.shape}")
    print(f"📋 Preprocessed data:\n{data_df}")
    
    cluster = model.predict(data_df)[0]
    print(f"🎯 Predicted cluster: {cluster}")
    return cluster

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/form')  # Changed from /fields to /form
def show_form():
    return render_template('form.html')

@app.route('/fields')  # Keep this for backward compatibility
def fields():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("📝 Received form submission")
    
    user_data = {
        'StudentLevel': request.form['StudentLevel'],
        'Discipline': request.form['Discipline'],
        'SessionLengthMin': request.form['SessionLengthMin'],
        'TotalPrompts': request.form['TotalPrompts'],
        'TaskType': request.form['TaskType'],
        'AI_AssistanceLevel': request.form['AI_AssistanceLevel'],
        'FinalOutcome': request.form['FinalOutcome'],
        'UsedAgain': request.form['UsedAgain'],
        'SatisfactionRating': request.form['SatisfactionRating']
    }
    print("📊 Form data received:", user_data)
    
    # Check for any empty or invalid values
    for key, value in user_data.items():
        if not value or value.strip() == '' or value == 'None':
            error_msg = f"Missing or invalid value for {key}: '{value}'"
            print(f"❌ Validation error: {error_msg}")
            return render_template('result.html', cluster=None, error=error_msg)
    
    try:
        cluster = predict_cluster(user_data)
        print(f"✅ Prediction successful! Cluster: {cluster}")
        return render_template('result.html', cluster=cluster)
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"❌ Prediction error: {error_msg}")
        print(f"📋 Full traceback:\n{traceback.format_exc()}")
        return render_template('result.html', cluster=None, error=error_msg)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('RENDER') != '1' 
    app.run(host='0.0.0.0', port=port, debug=debug_mode)