import pandas as pd
from settings import *
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import category_encoders as ce
#scale
from sklearn.preprocessing import StandardScaler
#classification models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
#accuracy scores and errors
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

import customtkinter as ctk

class FeedbackLabel(ctk.CTkFrame):
    def __init__(self , parent , content):
        super().__init__(parent , fg_color=OFF_WHITE)
        self.pack()

        normal_font = ctk.CTkFont('Calibri' , 18 )
        ctk.CTkLabel(self , image=CHECK , text = '').pack(side = 'left')
        ctk.CTkLabel(self , text = content , text_color= CHARCOAL , font=normal_font).pack(side = 'left' , padx = 4)

#scaler
scaler = StandardScaler()
#train and test variables
data = []
#columns dropped in test data
columns = []
# Create a SimpleImputer for numerical features (using median)
num_imputer = SimpleImputer(strategy='median')
# Create a SimpleImputer for categorical features (using most frequent)
cat_imputer = SimpleImputer(strategy='most_frequent')
#target encoder 
target_encoder = None
#one hot encoder
one_hot_encoder = None
#model
model = None


def preprocessing(file):
    #STEP 1 , DROPPING OF COLUMNS
    file = file.drop_duplicates()
    #drop columns dropped in train data
    file = file.drop(columns = columns , axis = 1)

    #STEP 2 , IMPUTATION
    #numerical columns
    numerical_cols = file.select_dtypes(include=['int64' , 'float64']).columns.tolist()
    #categorical columns
    cat_cols = file.select_dtypes(include=['object']).columns.tolist()
    #Impute missing values in categorical columns
    if len(numerical_cols) != 0 :
        file[numerical_cols] = num_imputer.transform(file[numerical_cols])
    # Impute missing values in categorical columns
    if len(cat_cols) != 0 :
        file[cat_cols] = cat_imputer.transform(file[cat_cols])

    #STEP 3 , TARGET ENCODING
    if target_encoder != None :
        file = target_encoder.transform(file)

    #STEP 4 , SCALING
    numerical_cols = file.select_dtypes(include=['int64' , 'float64']).columns.tolist()
    if len(numerical_cols) != 0 :
        file[numerical_cols] = scaler.transform(file[numerical_cols])

    #STEP 5 , ONE HOT ENCODING
    if one_hot_encoder != None :
        file = one_hot_encoder.transform(file)

    #STEP 6 , DROP NAN ROWS
    #due to the fact that some featues that the encoders have not seen before are empty
    nan_mask = file.isna().any(axis=1)
    file = file.dropna()

    return file , nan_mask


#scale
def scale(file):
    #numerical columns
    num_cols = file.select_dtypes(exclude = ['object']).columns
    if len(num_cols) != 0 :
        file[num_cols] = scaler.fit_transform(file[num_cols])

    return file

#categorical encoding
def categoricalEncoding(cardinality_limit , file , parent):
    global one_hot_encoder , target_encoder
    #categorical columns
    cat_cols = file.select_dtypes(include=['object']).columns
    FeedbackLabel(parent , "Cardinality Limit Received")
    high_cardinal_columns = []
    normal_cardinal_columns = []
    #segment categorical columns to high and low cardinal
    for cat_col in cat_cols:
        if file[cat_col].nunique() >= int(cardinality_limit) :
            high_cardinal_columns.append(cat_col)
        else:
            normal_cardinal_columns.append(cat_col)
    
    #target encoding
    target_encoder = ce.TargetEncoder(cols = high_cardinal_columns , handle_unknown='ignore')
    y_train = data[2]
    file = target_encoder.fit_transform(X=file ,y= y_train)
    #scale the file 
    file = scale(file)
    #one hot encoding
    one_hot_encoder = ce.OneHotEncoder(cols = normal_cardinal_columns , handle_unknown='ignore')
    file = one_hot_encoder.fit_transform(file)

    return file

#drop null columns
def dropNullColumns(file , percent , threshold , target_column , parent ):
    print(file.shape)
    columns_to_drop = []
    indices = np.where(percent > int(threshold) / 100 )[0]
    for index in indices :
        if file.columns.tolist()[index] != target_column :
            column = file.columns.tolist()[index]
            columns_to_drop.append(column)
            columns.append(column)
    
    if len(columns_to_drop) == 1 :
        file = file.drop(columns_to_drop[0] , axis = 1)
    elif len(columns_to_drop) > 1 :
        file = file.drop(columns_to_drop , axis = 1)

    FeedbackLabel(parent , f"Columns that are at least {threshold}% empty dropped")
    ctk.CTkLabel(parent , 
                    text= f"columns : {file.shape[1]} , rows {file.shape[0]}" ,
                    font = ctk.CTkFont('Calibri' , 18 , weight='bold'), 
                    text_color=CHARCOAL).pack(pady = 4)
    return  file

#data imputation
def dataImputation(file , parent):
    #numerical columns
    numerical_cols = file.select_dtypes(include=['int64' , 'float64']).columns.tolist()
    #categorical columns
    cat_cols = file.select_dtypes(include=['object']).columns.tolist()
    print(cat_cols , len(cat_cols))
    #Impute missing values in categorical columns
    if len(numerical_cols) != 0 :
        file[numerical_cols] = num_imputer.fit_transform(file[numerical_cols])
    # Impute missing values in categorical columns
    if len(cat_cols) != 0 :
        file[cat_cols] = cat_imputer.fit_transform(file[cat_cols])
    print(file)

    FeedbackLabel(parent , "Missing files have been imputed")

#train test split 
def trainTestSplit(file , train_percent , target):
    X = file.drop(columns=[target] , axis = 1)
    y = file[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                         test_size = 1- (int(train_percent) / 100) 
                                                         , random_state=42)

    return X_train, X_test, y_train, y_test

#create machine learning model
def createModel(model_name , X_train , parent , loading_tag , save_model_button):
    global model

    X_test , nan_mask = preprocessing(data[1])
    y_train = data[2]
    y_test = data[3][~nan_mask]
    #boolean to determine model type
    is_classifier = False
    #initialize model
    if model_name == "Random Forest Classifier":
        model = RandomForestClassifier()
        is_classifier = True
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor()
        is_classifier = False
    elif model_name == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
        is_classifier = True
    elif model_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor()
        is_classifier = False
    elif model_name == "Logistic Regression":
        model = LogisticRegression()
        is_classifier = True
    elif model_name == "Linear Regression":
        model = LinearRegression()
        is_classifier = False
    elif model_name == "Support Vector Classifier":
        model = SVC(kernel='linear')
        is_classifier = True
    elif model_name == "Support Vector Regressor":
        model = SVR(kernel='linear')
        is_classifier = False

    print(X_train)
    print(X_test)
    model.fit(X_train , y_train)
    predictions = model.predict(X_test)
    train_predictions = model.predict(X_train)

    FeedbackLabel(parent , "model fitted and predictions calculated")

    #make loading tag disappear , don't worry about it
    loading_tag.pack_forget()

    #scores and errors
    if is_classifier :
        test_accuracy = accuracy_score(predictions , y_test)
        train_accuracy = accuracy_score(train_predictions , y_train)
        FeedbackLabel(parent=parent , content= f"Test accuracy score : {np.around(test_accuracy , 2)}")
        FeedbackLabel(parent=parent , content= f"Train accuracy score : {np.around(train_accuracy , 2)}")
    else :
        test_mse = mean_squared_error(predictions , y_test)
        FeedbackLabel(parent=parent , content= f"Test Mean Squared Error : {np.around(test_mse , 2)}")
        train_mse = mean_squared_error(train_predictions , y_train)
        FeedbackLabel(parent=parent , content= f"Train Mean Squared Error : {np.around(train_mse , 2)}")
        test_mae = mean_absolute_error(predictions , y_test)
        FeedbackLabel(parent=parent , content= f"Test Mean Absolute Error : {np.around(test_mae , 2)}")
        train_mae = mean_absolute_error(train_predictions , y_train)
        FeedbackLabel(parent=parent , content= f"Train Mean Absolute Error : {np.around(train_mae , 2)}")

    save_model_button.pack(pady = 4)

def saveModel(model_name , file_name):
    #make folder to store output
    if 'Results' not in os.listdir('ML-No COde'):
        os.mkdir("ML-No COde/Results")

    index = len(os.listdir("ML-No COde/Results"))

    folder_name = str(index) + " " + file_name
    folder_name_present = True
    while(folder_name_present):
        if folder_name in os.listdir("ML-No COde/Results"):
            folder_name = str(index + 1) + " " + file_name
        else :
            folder_name_present = False
    
    os.mkdir(f"ML-No COde/Results/{folder_name}")

    #save files
    with open(f'ML-No COde/Results/{folder_name}/1 {model_name}' , 'wb') as file :
        pickle.dump(model , file)

    with open(f'ML-No COde/Results/{folder_name}/2 columns dropped' , 'wb') as file :
        pickle.dump(columns , file)

    with open(f'ML-No COde/Results/{folder_name}/3 scalar' , 'wb') as file :
        pickle.dump(scaler , file)

    with open(f'ML-No COde/Results/{folder_name}/4 Numerical Imputer' , 'wb') as file :
        pickle.dump(num_imputer , file)

    with open(f'ML-No COde/Results/{folder_name}/5 Categorical Imputer' , 'wb') as file :
        pickle.dump(cat_imputer , file)

    with open(f'ML-No COde/Results/{folder_name}/6 One Hot Encoder' , 'wb') as file :
        pickle.dump(one_hot_encoder , file)

    with open(f'ML-No COde/Results/{folder_name}/7 Target Encoder' , 'wb') as file :
        pickle.dump(target_encoder , file)
