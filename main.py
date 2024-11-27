from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import uvicorn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import sklearn
sklearn.set_config(transform_output="pandas")
from joblib import load


def transform_data(dataset):
    # Удаление пропусков и невозможных значений
    dataset.loc[dataset['max_power'] == ' bhp', 'max_power'] = np.nan
    dataset.loc[dataset['max_power'] == '0', 'max_power'] = np.nan
    # Перевод в числовой формат float/int
    dataset['mileage'] = dataset['mileage'].apply(lambda x: (float(x.split(' ')[0]) if x.split(' ')[1] == 'kmpl' else float(x.split(' ')[0])*1.4) # Переводим  km/kg в kmpl
                                                  if type(x)!=float
                                                  else x)
    dataset['engine'] = dataset['engine'].apply(lambda x: float(x.split(' ')[0]) if type(x)!=float else x)
    dataset['max_power'] = dataset['max_power'].apply(lambda x: float(x.split(' ')[0]) if type(x)!=float else x)
    dataset['engine'] = dataset['engine'].astype(int)
    dataset['name'] = dataset['name'].apply(lambda x: x.split(' ')[0])
    # Возвращение преобразованного датафрейма
    return dataset

def fill_NA(dataset, train_median_mileage, train_median_engine, train_median_max_power, train_median_seats):
    # Заполнение пропусков медианой трейна
    dataset['mileage'] = dataset['mileage'].fillna(train_median_mileage)
    dataset['engine'] = dataset['engine'].fillna(train_median_engine)
    dataset['max_power'] = dataset['max_power'].fillna(train_median_max_power)
    dataset['seats'] = dataset['seats'].fillna(train_median_seats)
    dataset['seats'] = dataset['seats'].astype('object')
    # Возвращение преобразованного датафрейма
    return dataset


app = FastAPI()

# Загрузка данных
df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv').dropna()
y_train = df_train['selling_price']
X_train = transform_data(df_train.drop(['selling_price', 'torque'], axis=1))

# Медианы трейна
train_median_mileage = X_train['mileage'].median()
train_median_engine = X_train['engine'].median()
train_median_max_power = X_train['max_power'].median()
train_median_seats = X_train['seats'].median()

# Заполнение медианами
X_train = fill_NA(X_train, train_median_mileage, train_median_engine, train_median_max_power, train_median_seats)

# Кодирование name
tgt_enc = ce.TargetEncoder(cols=['name'], smoothing=1)
tgt_enc.fit(X_train, y_train)
X_train = tgt_enc.transform(X_train)

# OHE
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_train = encoder.fit_transform(X_train.select_dtypes('object'))
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(list(X_train.select_dtypes('object').columns)))
X_train_encoded = pd.concat([X_train.select_dtypes(exclude='object').reset_index(drop=True)
                                , encoded_train_df.reset_index(drop=True)], axis=1)

# Скалирование
scaler = StandardScaler()
scaler.fit(X_train_encoded)
X_train_scaler = pd.DataFrame(scaler.transform(X_train_encoded), columns=X_train_encoded.columns)

# Загрузка модели
#model = Ridge(alpha=0.7)
#.fit(X_train_scaler, y_train)
model = load('model.pkl')

print('http://127.0.0.1:8000/docs')

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # Конвертация в датафрейм и удаление ненужных столбцов
    input_data = pd.DataFrame([item.dict()]).drop(['selling_price', 'torque'], axis=1)
    # Обработка
    data = transform_data(input_data)
    data = fill_NA(data, train_median_mileage, train_median_engine, train_median_max_power, train_median_seats)
    # Кодирование name
    data = tgt_enc.transform(data)
    # OHE
    encoded_data = encoder.transform(data.select_dtypes('object'))
    encoded_test_df = pd.DataFrame(encoded_data,
                                 columns=encoder.get_feature_names_out(data.select_dtypes('object').columns))
    data_cat_encoded = pd.concat(
      [data.select_dtypes(exclude='object').reset_index(drop=True), encoded_test_df.reset_index(drop=True)],
      axis=1)
    # Скалирование
    data_cat_encoded_scal = pd.DataFrame(scaler.transform(data_cat_encoded), columns=data_cat_encoded.columns)
    # Предсказания модели
    predicted_price = model.predict(data_cat_encoded_scal.values)
    # Возврат предсказанного значения
    return float(round(predicted_price[0], 2))


@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    # Конвертация в датафрейм и удаление ненужных столбцов
    input_data = items.dict()
    input_data = pd.DataFrame(input_data['objects'])
    input_data = input_data.drop(['selling_price', 'torque'], axis=1)
    # Обработка
    data = transform_data(input_data)
    data = fill_NA(data, train_median_mileage, train_median_engine, train_median_max_power, train_median_seats)
    # Кодирование name
    data = tgt_enc.transform(data)
    # OHE
    encoded_data = encoder.transform(data.select_dtypes('object'))
    encoded_test_df = pd.DataFrame(encoded_data,
                                   columns=encoder.get_feature_names_out(data.select_dtypes('object').columns))
    data_cat_encoded = pd.concat(
        [data.select_dtypes(exclude='object').reset_index(drop=True), encoded_test_df.reset_index(drop=True)],
        axis=1)
    # Скалирование
    data_cat_encoded_scal = pd.DataFrame(scaler.transform(data_cat_encoded), columns=data_cat_encoded.columns)
    # Предсказания модели
    predicted_prices = model.predict(data_cat_encoded_scal.values)
    # Возврат списка предсказанных значений
    return [float(round(price, 2)) for price in predicted_prices]

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)