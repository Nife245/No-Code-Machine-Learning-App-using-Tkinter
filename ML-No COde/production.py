import os
import pickle

#note that the file that is to be predicted must be a dataframe 

#folder that contains the saved machine learning algorithm and other files
folder = "ML-No COde\Results/0 melb_data"

#load model
def loadModel(folder):
    data = os.listdir(folder)
    model = pickle.load(open(folder + '/' + data[0] , 'rb'))
    dropped_columns = pickle.load(open(folder + '/' + data[1] , 'rb'))
    scaler = pickle.load(open(folder + '/' + data[2] , 'rb'))
    numerical_imputer = pickle.load(open(folder + '/' + data[3] , 'rb'))
    categorical_imputer = pickle.load(open(folder + '/' + data[4] , 'rb'))
    one_hot_encoder = pickle.load(open(folder + '/' + data[5] , 'rb'))
    target_encoder = pickle.load(open(folder + '/' + data[6] , 'rb'))

    return (model , dropped_columns , scaler , numerical_imputer , categorical_imputer ,
        one_hot_encoder , target_encoder)

#preprocessing
def preProcess(file):
    (model , dropped_columns , scaler , numerical_imputer , categorical_imputer ,
        one_hot_encoder , target_encoder) = loadModel(folder=folder)
    
    #STEP 1 , DROPPING OF COLUMNS
    file = file.drop_duplicates()
    #drop columns dropped in train data
    print(dropped_columns)
    file = file.drop(columns = dropped_columns , axis = 1)

    #STEP 2 , IMPUTATION
    #numerical columns
    numerical_cols = file.select_dtypes(include=['int64' , 'float64']).columns.tolist()
    #categorical columns
    cat_cols = file.select_dtypes(include=['object']).columns.tolist()
    #Impute missing values in categorical columns
    file[numerical_cols] = numerical_imputer.transform(file[numerical_cols])
    # Impute missing values in categorical columns
    file[cat_cols] = categorical_imputer.transform(file[cat_cols])

    #STEP 3 , TARGET ENCODING
    file = target_encoder.transform(file)

    #STEP 4 , SCALING
    numerical_cols = file.select_dtypes(include=['int64' , 'float64']).columns.tolist()
    file[numerical_cols] = scaler.transform(file[numerical_cols])

    #STEP 5 , ONE HOT ENCODING
    file = one_hot_encoder.transform(file)

    predictions = model.predict(file)

    return predictions


