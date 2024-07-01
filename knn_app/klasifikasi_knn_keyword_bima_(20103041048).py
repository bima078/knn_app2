# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import streamlit as st

# Step 1: Read the Data
try:
    data = pd.read_csv('D:\Kuliah\Semester 8\Machine Learning\knn_app\data\Book1.csv')
    print("Data loaded successfully")
except Exception as e:
    print("Error loading data:", e)
    raise e

data

data.info()

data = data.astype(float)

data.isnull().sum()


#Menghapus atribut keyword difficulty
columnDrop = ['Keyword Difficulty']
data = data.drop(columnDrop, axis = 1)

#Mengecek kolom
data.info()

#Mengecek atribut apakah berthasil di drop atau tidak
duplicateData = data.duplicated()
data[duplicateData]

#Menentukan target klasifikasi
data['Tingkat Trending'].value_counts()

x = data.drop("Tingkat Trending", axis = 1).values
y = data.iloc[:,-1]

from imblearn.over_sampling import SMOTE

#Oversampling
smote = SMOTE(random_state = 42)
xSmote_resampled, ySmote_resampled = smote.fit_resample(x, y)

newDF1 = pd.DataFrame(data = y)

newDF1.value_counts().plot(kind = 'bar', figsize = (10, 6), color = ['green', 'blue'])

newDF2 = pd.DataFrame(data = ySmote_resampled)

newDF2.value_counts().plot(kind = 'bar', figsize = (10, 6), color = ['green', 'blue'])

newDF1 = pd.DataFrame(data = y)
newDF1.value_counts()

newDF2 = pd.DataFrame(data = ySmote_resampled)
newDF2.value_counts()

data.describe()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
xSmote_resampled_normal = scaler.fit_transform(xSmote_resampled)
# len(xSmote_resampled_normal)

dfCek1 = pd.DataFrame(xSmote_resampled_normal)
dfCek1.describe()

from sklearn.model_selection import train_test_split

# Membagi fitur dan target menjadi data train dan test (untuk yang oversampling)
xTrain, xTest, yTrain, yTest = train_test_split(xSmote_resampled, ySmote_resampled, test_size = 0.4, random_state = 42, stratify = ySmote_resampled)

# Membagi fitur dan target menjadi data train dan test (untuk yang oversampling)
xTrain_normal, xTest_normal, yTrain_normal, yTest_normal = train_test_split(xSmote_resampled_normal, ySmote_resampled, test_size = 0.4, random_state = 42, stratify = ySmote_resampled)


from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix, precision_score

def evaluation (yTest, yPred):
  accTest = accuracy_score(yTest, yPred)
  rclTest = recall_score(yTest, yPred, average = 'weighted')
  f1Test = f1_score(yTest, yPred, average = 'weighted')
  psTest = precision_score(yTest, yPred, average = 'weighted')

  metric_dict = {'accuracy Score' : round (accTest, 3),
               'recall Score' : round (rclTest, 3),
               'f1 Score' : round (f1Test, 3),
               'Precision Score' : round (psTest, 3)
               }

  return print(metric_dict)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(xTrain, yTrain)

yPred_knn = knn_model.predict(xTest)

#Evaluasi KNN Model
print("K-Nearest Neighbors (KNN) Model: ")
accuracy_knn_smote = round(accuracy_score(yTest, yPred_knn), 3)

print("Accuracy: ", accuracy_knn_smote)
print("Classification Report: ")
print(classification_report(yTest, yPred_knn))

confMatrix = confusion_matrix(yTest, yPred_knn)


modelComp1 = pd.DataFrame({'Model' : ['K-Nearest Neighbour'], 'Accuracy': [accuracy_knn_smote*100, ]})

modelComp1.head()

# Simpan model KNN ke file menggunakan pickle
with open('model\knn_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)

# Muat model KNN dari file menggunakan pickle
with open('model\knn_model.pkl', 'rb') as file:
    loaded_knn_model = pickle.load(file)

# STREAMLIT
st.set_page_config(page_title="Klasifikasi Keyword Pencarian Produk PMB Toys Menggunakan KNN")

st.title("Klasifikasi Keyword Pencarian Produk PMB Toys Menggunakan   ")
st.write(f"**_Model's Accuracy_** :  :green[**{accuracy_knn_smote}**]% (:red[_Do not copy outright_])")
st.write("")

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

with tab1:
    st.header("Single Prediction")
    st.write("Masukkan nilai untuk melakukan prediksi tunggal.")

    volume = st.number_input(label="Volume", min_value=float(data['Volume'].min()), max_value=float(data['Volume'].max()))
    trend_apr = st.number_input(label="Trend April'23", min_value=float(data["Trend April'23"].min()), max_value=float(data["Trend April'23"].max()))
    trend_may = st.number_input(label="Trend Mei'23", min_value=float(data["Trend Mei'23"].min()), max_value=float(data["Trend Mei'23"].max()))
    trend_jun = st.number_input(label="Trend Juni'23", min_value=float(data["Trend Juni'23"].min()), max_value=float(data["Trend Juni'23"].max()))
    trend_jul = st.number_input(label="Trend Juli'23", min_value=float(data["Trend Juli'23"].min()), max_value=float(data["Trend Juli'23"].max()))
    trend_aug = st.number_input(label="Trend Agu'23", min_value=float(data["Trend Agu'23"].min()), max_value=float(data["Trend Agu'23"].max()))
    trend_sep = st.number_input(label="Trend Sep'23", min_value=float(data["Trend Sep'23"].min()), max_value=float(data["Trend Sep'23"].max()))
    trend_oct = st.number_input(label="Trend Okt'23", min_value=float(data["Trend Okt'23"].min()), max_value=float(data["Trend Okt'23"].max()))
    trend_nov = st.number_input(label="Trend Nov'23", min_value=float(data["Trend Nov'23"].min()), max_value=float(data["Trend Nov'23"].max()))
    trend_dec = st.number_input(label="Trend Dec'23", min_value=float(data["Trend Dec'23"].min()), max_value=float(data["Trend Dec'23"].max()))
    trend_jan = st.number_input(label="Trend Jan'24", min_value=float(data["Trend Jan'24"].min()), max_value=float(data["Trend Jan'24"].max()))
    trend_feb = st.number_input(label="Trend Feb'24", min_value=float(data["Trend Feb'24"].min()), max_value=float(data["Trend Feb'24"].max()))
    trend_mar = st.number_input(label="Trend Mar'24", min_value=float(data["Trend Mar'24"].min()), max_value=float(data["Trend Mar'24"].max()))

    input_array = [volume, trend_apr, trend_may, trend_jun, trend_jul, trend_aug, trend_sep, trend_oct, trend_nov, trend_dec, trend_jan, trend_feb, trend_mar]

    if st.button("Predict"):
        class_value = loaded_knn_model.predict([input_array])
        st.subheader("Predicted Class")
        st.write(f"Tingkat Trending: {class_value[0]}")

with tab2:
    st.header("Multiple Predictions from CSV")
    st.write("Upload file CSV untuk melakukan prediksi multiple.")

    uploaded_file = st.file_uploader("Choose a CSV file")

    if uploaded_file is not None:
        data_multi_predict = pd.read_csv(uploaded_file)
        predictions = loaded_knn_model.predict(data_multi_predict)

        st.subheader("Prediction Results")
        st.write(data_multi_predict.assign(predicted_class=predictions))

        csv = data_multi_predict.to_csv(index=False)
        st.download_button(label="Download Prediction Results", data=csv, file_name='prediction_results.csv')
