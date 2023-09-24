from flask import Flask, render_template, url_for, flash, redirect
import joblib
from flask import request
import numpy as np
import pandas as pd
import tensorflow
import cv2

import os
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__, template_folder='template')

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

from tensorflow.keras.models import load_model

malaria_model = load_model('Models/Malaria.h5')
pneumonia_model = load_model('Models/PNEUMONIA_classifier.h5')
brain_tumor_model = load_model('Models/BrainTumor.h5')
eye_disease_model = load_model('Models/Eye_disease_TF.h5')


# home page

# @app.route('/')
# def home():
#  return render_template('index.html')


@app.route('/malaria', methods=['POST', 'GET'])
def malaria():
    if request.method == 'GET':
        return render_template('malaria.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            img = cv2.imread(full_name)
            resize = tf.image.resize(img, (256, 256))
            result = malaria_model.predict(np.expand_dims(resize / 255, 0))
            label = "Uninfected" if result[0][0] > 0.5 else "Parasite"
            return render_template('malaria_prediction.html', image_file_name=file.filename, label=label)
        except:
            flash("Please select the image first !!", "danger")
            return redirect(url_for("Malaria"))


@app.route('/pneumonia', methods=['POST', 'GET'])
def pneumonia():
    if request.method == 'GET':
        return render_template('pneumonia.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            img = cv2.imread(full_name)
            resize = tf.image.resize(img, (256, 256))
            result = pneumonia_model.predict(np.expand_dims(resize / 255, 0))
            label = "PNEUMONIA" if result[0][0] > 0.5 else "NORMAL"
            return render_template('pneumonia_prediction.html', image_file_name=file.filename, label=label)
        except:
            flash("Please select the image first !!", "danger")
            return redirect(url_for("Pneumonia"))


@app.route('/brain_tumor', methods=['POST', 'GET'])
def brain_tumor():
    if request.method == 'GET':
        return render_template('brain_tumor.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            img = cv2.imread(full_name)
            resize = tf.image.resize(img, (256, 256))
            result = brain_tumor_model.predict(np.expand_dims(resize / 255, 0))
            label = "Brain Tumor" if result[0][0] > 0.5 else "No Brain Tumor"
            return render_template('brain_tumor_prediction.html', image_file_name=file.filename, label=label)
        except:
            flash("Please select the image first !!", "danger")
            return redirect(url_for("Brain_Tumor"))


@app.route('/eye_disease', methods=['POST', 'GET'])
def eye_disease():
    if request.method == 'GET':
        return render_template('eye_disease.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            img = cv2.imread(full_name)
            resize = tf.image.resize(img, (224, 224))

            eye_disease_labels = {0: "Cataract", 1: "Diabetic_retinopathy", 2: "Glaucoma", 3: "Normal"}

            result = eye_disease_model.predict(np.expand_dims(resize / 255, 0))
            label = eye_disease_labels[np.argmax(result)]
            return render_template('eye_disease_prediction.html', image_file_name=file.filename, label=label)
        except:
            flash("Please select the image first !!", "danger")
            return redirect(url_for("Eye_Disease"))


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/cancer")
def cancer():
    return render_template("cancer.html")


@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


@app.route("/heart")
def heart():
    return render_template("heart.html")


@app.route("/liver")
def liver():
    return render_template("liver.html")


@app.route("/kidney")
def kidney():
    return render_template("kidney.html")


@app.route("/Malaria")
def Malaria():
    return render_template("malaria.html")


@app.route("/Pneumonia")
def Pneumonia():
    return render_template("pneumonia.html")


@app.route("/Brain_Tumor")
def Brain_Tumor():
    return render_template("brain_tumor.html")


@app.route("/Eye_Disease")
def Eye_Disease():
    return render_template("eye_disease.html")


@app.route("/transcription_classifier")
def transcription_classifier():
    return render_template("medical_transcription_classification_home.html")



@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['chest pain type'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['serum cholestoral in mg/dl'])
        fbs = float(request.form['fasting blood sugar > 120 mg/dl'])
        restecg = float(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        test = pd.DataFrame(columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                                     'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        test.loc[0] = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        heart_model = joblib.load('Models/heart_model')
        result = heart_model.predict(test)

        if (int(result[0]) == 1):
            prediction = 'Sorry ! you have the heart disease'
        else:
            prediction = 'Congrats ! you are Healthy'

        return render_template('result.html', prediction=prediction)


@app.route('/predict_diabetics', methods=['POST'])
def predict_diabetics():
    if request.method == 'POST':
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])

        test = pd.DataFrame(columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                                     'BMI', 'DiabetesPedigreeFunction', 'Age'])
        test.loc[0] = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diabetic_model = joblib.load('Models/diabetic_model')
        result = diabetic_model.predict(test)

        if (int(result[0]) == 1):
            prediction = 'Sorry ! you have the diabetics'
        else:
            prediction = 'Congrats ! you are Healthy'

        return render_template('result.html', prediction=prediction)


@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    if request.method == 'POST':
        Radius_mean = float(request.form['Radius_mean'])
        Texture_mean = float(request.form['Texture_mean'])
        Perimeter_mean = float(request.form['Perimeter_mean'])
        Area_mean = float(request.form['Area_mean'])
        Smoothness_mean = float(request.form['Smoothness_mean'])
        Compactness_mean = float(request.form['Compactness_mean'])
        Concavity_mean = float(request.form['Concavity_mean'])
        concave_points_mean = float(request.form['concave points_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
        radius_se = float(request.form['radius_se'])
        texture_se = float(request.form['texture_se'])
        perimeter_se = float(request.form['perimeter_se'])
        area_se = float(request.form['area_se'])
        smoothness_se = float(request.form['smoothness_se'])
        compactness_se = float(request.form['compactness_se'])
        concavity_se = float(request.form['concavity_se'])
        concave_points_se = float(request.form['concave points_se'])
        symmetry_se = float(request.form['symmetry_se'])
        fractal_dimension_se = float(request.form['fractal_dimension_se'])
        radius_worst = float(request.form['radius_worst'])
        texture_worst = float(request.form['texture_worst'])
        perimeter_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        smoothness_worst = float(request.form['smoothness_worst'])
        compactness_worst = float(request.form['compactness_worst'])
        concavity_worst = float(request.form['concavity_worst'])
        concave_points_worst = float(request.form['concave points_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

        test = pd.DataFrame(columns=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                                     'smoothness_mean', 'compactness_mean', 'concavity_mean',
                                     'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                                     'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                                     'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                                     'fractal_dimension_se', 'radius_worst', 'texture_worst',
                                     'perimeter_worst', 'area_worst', 'smoothness_worst',
                                     'compactness_worst', 'concavity_worst', 'concave points_worst',
                                     'symmetry_worst', 'fractal_dimension_worst'])
        test.loc[0] = [Radius_mean, Texture_mean, Perimeter_mean, Area_mean, Smoothness_mean, Compactness_mean,
                       Concavity_mean, concave_points_mean,
                       symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                       smoothness_se, compactness_se,
                       concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
                       perimeter_worst, area_worst,
                       smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst,
                       fractal_dimension_worst]
        breast_cancer_model = joblib.load('Models/breast_cancer_model')
        result = breast_cancer_model.predict(test)

        if (int(result[0]) == 1):
            prediction = 'Sorry ! you have the cancer disease'
        else:
            prediction = 'Congrats ! you are Healthy'

        return render_template('result.html', prediction=prediction)


@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():
    if request.method == 'POST':
        age = float(request.form['age'])
        bp = float(request.form['bp'])
        sg = float(request.form['sg'])
        al = float(request.form['al'])
        su = float(request.form['su'])
        rbc = str(request.form['rbc'])
        pc = str(request.form['pc'])
        pcc = str(request.form['pcc'])
        ba = str(request.form['ba'])
        bgr = float(request.form['bgr'])
        bu = float(request.form['bu'])
        sc = float(request.form['sc'])
        sod = float(request.form['sod'])
        pot = float(request.form['pot'])
        hemo = float(request.form['hemo'])
        pcv = float(request.form['pcv'])
        wc = float(request.form['wc'])
        rc = float(request.form['rc'])
        htn = str(request.form['htn'])
        dm = str(request.form['dm'])
        cad = str(request.form['cad'])
        appet = str(request.form['appet'])
        pe = str(request.form['pe'])
        ane = str(request.form['ane'])

        test = pd.DataFrame(columns=['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
                                     'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
                                     'appet', 'pe', 'ane'])
        test.loc[0] = [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,
                       appet, pe, ane]
        kidney_model = joblib.load('Models/kidney_model')
        result = kidney_model.predict(test)

        if (int(result[0]) == 1):
            prediction = 'Sorry ! you have the kidney disease'
        else:
            prediction = 'Congrats ! you are Healthy'

        return render_template('result.html', prediction=prediction)


@app.route('/predict_liver', methods=['POST'])
def predict_liver():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Gender = int(request.form['Gender'])
        Total_Bilirubin = float(request.form['Total_Bilirubin'])
        Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
        Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
        Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
        Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
        Total_Protiens = float(request.form['Total_Protiens'])
        Albumin = float(request.form['Albumin'])
        Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])

        test = pd.DataFrame(columns=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                                     'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
                                     'Albumin',
                                     'Albumin_and_Globulin_Ratio'])
        test.loc[0] = [Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase,
                       Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]
        liver_model = joblib.load('Models/liver_model')
        result = liver_model.predict(test)

        if (int(result[0]) == 1):
            prediction = 'Sorry ! you have the liver disease'
        else:
            prediction = 'Congrats ! you are Healthy'

        return render_template('result.html', prediction=prediction)


@app.route('/predict_transcription_classification', methods=['POST'])
def predict_transcription_classification():
    if request.method == 'POST':
        medical_transcription_classifier = joblib.load('Models/medical_transcriptions_classifier')
        transcriptions = request.form['message']
        data = [transcriptions]
        model_prediction = medical_transcription_classifier.predict(data)

        medical_transcription_labels = {0: "Cardiovascular / Pulmonary", 1: "Orthopedic", 2: "Surgery"}

        prediction = medical_transcription_labels[model_prediction[0]]

        return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
