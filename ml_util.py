# Import dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor


def clean_car_data(car_df):
        car_df['Levy'] = (car_df['Levy'].replace('-','0')).astype(int)
        car_df['Make model'] = car_df['Manufacturer'] + ' ' + car_df['Model']
        car_df['Mileage'] = car_df['Mileage'].str.replace(' km', '', regex=False).astype(int)
        miles_per_km = 0.621371
        car_df['Mileage (mi)'] = car_df['Mileage']*miles_per_km
        car_df['Mileage (mi)'] = car_df['Mileage (mi)'].round(0).astype(int)
        car_df = car_df.drop(columns=['ID', 'Manufacturer', 'Model', 'Mileage', 'Doors', 'Wheel'])
        car_df.to_csv('clean_car_data.csv', index=False)
        return car_df
    
    
def encode_categorical_data(X_train, X_test):
    #train encoder for categorical data
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.set_output(transform="pandas")
    columns_to_encode = ['Category', 'Leather interior', 'Fuel type', 'Engine volume', 'Gear box type', 'Drive wheels', 'Make model', 'Color']
    ohe.fit(X_train[columns_to_encode])
    X_train_ohe = ohe.transform(X_train[columns_to_encode]).reset_index()
    X_test_ohe = ohe.transform(X_test[columns_to_encode]).reset_index()
    ohe.get_feature_names_out()
    return X_train_ohe, X_test_ohe


def scale_numeric_data(X_train, X_test):
    # isolate numeric data for X
    X_train_numeric_df = X_train.select_dtypes(include=['number'])
    X_test_numeric_df = X_test.select_dtypes(include=['number'])
    # scale dataset
    scaler = StandardScaler()
    scaler.fit(X_train_numeric_df)
    X_train_numeric_df_scaled = scaler.transform(X_train_numeric_df)
    X_test_numeric_df_scaled = scaler.transform(X_test_numeric_df)
    X_train_numeric_df_scaled = pd.DataFrame(X_train_numeric_df_scaled, columns=
                                             ['Levy', 'Prod. year', 'Cylinders', 'Airbags','Mileage (mi)'])
    X_test_numeric_df_scaled = pd.DataFrame(X_test_numeric_df_scaled, columns=
                                            ['Levy', 'Prod. year', 'Cylinders', 'Airbags','Mileage (mi)'])
    return X_train_numeric_df_scaled, X_test_numeric_df_scaled


def preprocess_car_data(car_df):
    clean_df = clean_car_data(car_df)
    X = clean_df.drop(columns='Price')
    y = clean_df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train_ohe, X_test_ohe = encode_categorical_data(X_train, X_test)
    X_train_numeric_df_scaled, X_test_numeric_df_scaled = scale_numeric_data(X_train, X_test)
    X_train = pd.concat([X_train_numeric_df_scaled, X_train_ohe], axis='columns')
    X_test = pd.concat([X_test_numeric_df_scaled, X_test_ohe], axis='columns')
    return X_train, X_test, y_train, y_test


def check_model_accuracy(car_df, models_to_test):
    X_train, X_test, y_train, y_test = preprocess_car_data(car_df)
    for model in models_to_test:
        model_name = type(model).__name__
        model.fit(X_train, y_train)
        print(f"{model_name} Train Score: {model.score(X_train, y_train)}")
        print(f"{model_name} Test Score: {model.score(X_test, y_test)}")
        print("-"*50)