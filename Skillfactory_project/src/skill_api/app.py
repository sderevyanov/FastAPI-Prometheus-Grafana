import json
import logging
import os

import joblib
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb
from tqdm import tqdm
from fastapi.responses import JSONResponse
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway, generate_latest

from .config import path_logs, path_scaler, path_models
from .data_prepare import outlier_low, outlier_high
from .schema import ModelTrain, ModelOutp, Wine, Rating, feature_names
import warnings

warnings.filterwarnings("ignore")

# Создадим логирование
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
path = path_logs()
filename = os.path.join(path, 'API_log_' +
                        str(datetime.now().year) +
                        str(datetime.now().strftime("%m")) +
                        str(datetime.now().strftime("%d")) +
                        '.log')
fh = logging.FileHandler(filename=filename)
formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
)
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)  # Exporting logs to the screen
logger.addHandler(fh)  # Exporting logs to a file

# Сделаем оформление UI для FastApi
app = FastAPI(title='Machine learning API',
              description='Creating machine learning models, refit models',
              version='0.0.1',
              contact={
                  "name": "Sergey Derevyanov",
                  "email": "derevyanov@mail.ru"},
              license_info={
                  "name": "Apache 2.0",
                  "url": "https://www.apache.org/licenses/LICENSE-2.0.html"},
              openapi_tags=[{"name": "check", "description": "Check html response."},
                            {"name": "train", "description": "Operations with training models."},
                            {"name": "forecast", "description": "Operations with forecasting"}])

# Включим CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------------------------------
# Зададим пути для сохранения моделей и моделей нормализации данных

path_m = path_models()
path_sc = path_scaler()

# Зададим параметры для pushgateway 
registry = CollectorRegistry()
g = Gauge('Wine_model_metric_Accuracy', 'Metric after train Wine model', registry=registry)

# ---------------------------------------------------------------------------------------------------


@app.get("/", name='Check HTMLResponse', tags=["check"])
def root():
    return HTMLResponse("<b>Machine learning API</b>")


@app.post('/model_train/', response_model=ModelOutp, name='Train model', tags=["train"])
async def train_model(data_js: ModelTrain):
    """
    Создание модели машинного обучения с помощью алгоритма CatBoostRegressor.
    """
    global path_m, path_sc

    start_time = datetime.now()

    # Загрузим данные из файла .json
    try:
        data = json.loads(data_js.json())
        logger.info('Данные загружены')
    except:
        logger.error('Данные неверного формата, ошибка в исходном файле')
        logger.info('Завершено')
        return {'Training_time': str(datetime.now() - start_time),
                'Status': str('Error'),
                'Message': str('Данные неверного формата, ошибка в исходном файле'),
                'accuracy_score': '0',
                'f1_score': '0'}

    # Создадим датафрейм
    df = pd.DataFrame(data['Data'])

    # Преобразуем типы строк на float
    columns_to_float = df.iloc[:, :-1].columns.to_list()
    for col in tqdm(columns_to_float):
        df[col] = df[col].astype(float)

    # Удалим выбросы
    cols_del_outliers = df.drop('quality', axis=1).columns.tolist()
    for col in cols_del_outliers:
        logger.info('Feature name: {}'.format(col))
        start_len = len(df)
        logger.info('Num rows before removal: {}'.format(start_len))
        df = df.loc[df[col].between(outlier_low(df[col]), outlier_high(df[col]), inclusive='both')]
        logger.info('Num rows after removal: {}'.format(len(df)))
        logger.info('Num deleted outliers: {}'.format(start_len - len(df)))

    # Разделим целевую переменную на 2 класса: 0 (bad wine) - если оценка < 6, 1 (good wine) - если оценка >=6
    df['quality_class'] = df['quality'].apply(lambda x: int(0) if x < 6 else int(1))

    # Разделим датасет на train и test
    X = df.drop(['quality', 'quality_class'], axis=1)
    y = df.quality_class

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1, stratify=y)

    # Масштабирование данных
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[feature_names] = scaler.fit_transform(
        X_train_scaled[feature_names])

    X_test_scaled[feature_names] = scaler.transform(
        X_test_scaled[feature_names])

    # Создадим датасеты для обучения lgb
    lgb_train = lgb.Dataset(X_train_scaled, y_train)
    lgb_eval = lgb.Dataset(X_test_scaled, y_test, reference=lgb_train)

    # Зададим параметры модели
    params = {
        'task': 'train',
        'application': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'tree_learner': 'serial',
        'metric': ['binary_logloss', 'auc'],
        'max_bin': 255,
    }

    # Зададим дату старта обучения
    logger.info('Запуск обучения модели.')
    t_train_start = datetime.now()

    # Обучим модель
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=10,
                    verbose_eval=False
                    )

    # Посчитаем время затраченное на обучение
    time_fit_all = datetime.now() - t_train_start
    logger.info(f'Обучение модели завершено. Время обучения: {time_fit_all}')

    # Спрогнозируем результаты
    pred_y = gbm.predict(X_test_scaled)
    pred_classes = np.where(pred_y > 0.5, 1, 0)

    logger.info('Сохраняем обученную модель и модель нормализации данных.')
    # Сохраним модель
    filename_m = os.path.join(path_m, "Wine_model" + '.joblib')
    joblib.dump(gbm, open(filename_m, 'wb'))
    # Сохраним модель нормализации данных
    filename_sc = os.path.join(path_sc, "Wine_scaler" + '.joblib')
    joblib.dump(scaler, open(filename_sc, 'wb'))

    logger.info('Модель и модель нормализации данных, сохранены.')
    logger.info('Обучение завершено.')

    # Передадим метрики в prometheus через push_to_gateway
    g.set(accuracy_score(y_test, pred_classes))
    push_to_gateway('localhost:9091', job='batchA', registry=registry)

    return {'Training_time': str(datetime.now() - start_time),
            'Status': str('Success'),
            'Message': str('Обучение модели прошло успешно.'),
            'accuracy_score': str(f"{accuracy_score(y_test, pred_classes):0.5f}"),
            'f1_score': str(f"{f1_score(y_test, pred_classes):0.5f}")
            }


@app.post('/model_refit/', response_model=ModelOutp, name='Refit model', tags=["train"])
async def train_model(data_js: ModelTrain):
    global path_m, path_sc

    # Загрузим модель из хранилища
    try:
        filename_m = os.path.join(path_m, "Wine_model" + ".joblib")
        with open(filename_m, 'rb') as f:
            model = joblib.load(f)
    except FileNotFoundError as f:
        logger.error('Модель {0} не существует.'.format("Wine_model" + ".joblib"))
        logger.info('Завершено')
        return JSONResponse(status_code=418,
                            content=str('Модель {0} не существует.'.format("Wine_model" + ".joblib")))
    logger.info('Модель загружена')

    # Загрузим модель нормализации данных из хранилища
    try:
        filename_sc = os.path.join(path_sc, "Wine_scaler" + ".joblib")
        with open(filename_sc, 'rb') as d:
            scaler = joblib.load(d)
    except FileNotFoundError as f:
        logger.error('Модель нормализации данных {0} не существует.'.format("Wine_scaler" + ".joblib"))
        logger.info('Завершено')
        return JSONResponse(status_code=418,
                            content=str('Модель {0} не существует.'.format("Wine_scaler" + ".joblib")))
    logger.info('Модель нормализации данных загружена')

    start_time = datetime.now()

    # Загрузим данные из файла .json
    try:
        data = json.loads(data_js.json())
        logger.info('Данные загружены')
    except:
        logger.error('Данные неверного формата, ошибка в исходном файле')
        logger.info('Завершено')
        return {'Training_time': str(datetime.now() - start_time),
                'Status': str('Error'),
                'Message': str('Данные неверного формата, ошибка в исходном файле'),
                'MAE': '0',
                'MAPE': '0'}

    # Создадим датафрейм
    df = pd.DataFrame(data['Data'])

    # Преобразуем типы строк на float
    columns_to_float = df.iloc[:, :-1].columns.to_list()
    for col in tqdm(columns_to_float):
        df[col] = df[col].astype(float)

    # Удалим выбросы
    cols_del_outliers = df.drop('quality', axis=1).columns.tolist()
    for col in cols_del_outliers:
        logger.info('Feature name: {}'.format(col))
        start_len = len(df)
        logger.info('Num rows before removal: {}'.format(start_len))
        df = df.loc[df[col].between(outlier_low(df[col]), outlier_high(df[col]), inclusive='both')]
        logger.info('Num rows after removal: {}'.format(len(df)))
        logger.info('Num deleted outliers: {}'.format(start_len - len(df)))

    # Разделим целевую переменную на 2 класса: 0 (bad wine) - если оценка < 6, 1 (good wine) - если оценка >=6
    df['quality_class'] = df['quality'].apply(lambda x: int(0) if x < 6 else int(1))

    # Разделим датасет на train и test
    X = df.drop(['quality', 'quality_class'], axis=1)
    y = df.quality_class

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1, stratify=y)

    # Масштабирование данных
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[feature_names] = scaler.fit_transform(
        X_train_scaled[feature_names])

    X_test_scaled[feature_names] = scaler.transform(
        X_test_scaled[feature_names])

    # Создадим датасеты для обучения lgb
    lgb_train = lgb.Dataset(X_train_scaled, y_train)
    lgb_eval = lgb.Dataset(X_test_scaled, y_test, reference=lgb_train)

    # Зададим параметры модели

    params = {
        'task': 'train',
        'application': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'tree_learner': 'serial',
        'metric': ['binary_logloss', 'auc'],
        'max_bin': 255,
    }

    # Зададим дату старта обучения
    logger.info('Запуск обучения модели.')
    t_train_start = datetime.now()

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    init_model=model,
                    early_stopping_rounds=10,
                    verbose_eval=False,
                    keep_training_booster=True)

    # Посчитаем время затраченное на обучение
    time_fit_all = datetime.now() - t_train_start
    logger.info(f'Обучение модели завершено. Время обучения: {time_fit_all}')

    # Спрогнозируем результаты
    pred_y = gbm.predict(X_test_scaled)
    pred_classes = np.where(pred_y > 0.5, 1, 0)

    logger.info('Сохраняем обученную модель и модель нормализации данных.')
    # Сохраним модель
    filename_m = os.path.join(path_m, "Wine_model" + '.joblib')
    joblib.dump(gbm, open(filename_m, 'wb'))
    # Сохраним модель нормализации данных
    filename_sc = os.path.join(path_sc, "Wine_scaler" + '.joblib')
    joblib.dump(scaler, open(filename_sc, 'wb'))

    logger.info('Модель и модель нормализации данных, сохранены.')
    logger.info('Обучение завершено.')

    # Передадим метрики в prometheus через push_to_gateway
    g.set(accuracy_score(y_test, pred_classes))  # Set to a given value
    push_to_gateway('localhost:9091', job='batchA', registry=registry)

    return {'Training_time': str(datetime.now() - start_time),
            'Status': str('Success'),
            'Message': str('Обучение модели прошло успешно.'),
            'accuracy_score': str(f"{accuracy_score(y_test, pred_classes):0.5f}"),
            'f1_score': str(f"{f1_score(y_test, pred_classes):0.5f}")
            }


@app.post("/predict", response_model=Rating, name='Predict class of wine', tags=["forecast"])
def predict(response: Response, sample: Wine):
    global path_m, path_sc

    # Загрузим модель из хранилища
    try:
        filename_m = os.path.join(path_m, "Wine_model" + ".joblib")
        with open(filename_m, 'rb') as f:
            model = joblib.load(f)
    except FileNotFoundError as f:
        logger.error('Модель {0} не существует.'.format("Wine_model" + ".joblib"))
        logger.info('Завершено')
        return JSONResponse(status_code=418,
                            content=str('Модель {0} не существует.'.format("Wine_model" + ".joblib")))
    logger.info('Модель загружена')

    # Загрузим модель нормализации данных из хранилища
    try:
        filename_sc = os.path.join(path_sc, "Wine_scaler" + ".joblib")
        with open(filename_sc, 'rb') as d:
            scaler = joblib.load(d)
    except FileNotFoundError as f:
        logger.error('Модель нормализации данных {0} не существует.'.format("Wine_scaler" + ".joblib"))
        logger.info('Завершено')
        return JSONResponse(status_code=418,
                            content=str('Модель {0} не существует.'.format("Wine_scaler" + ".joblib")))
    logger.info('Модель нормализации данных загружена')

    logger.info('Прогнозирование запущено.')
    sample_dict = sample.dict()
    features = np.array([sample_dict[f] for f in feature_names]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    prediction = np.where(prediction > 0.5, 1, 0)
    response.headers["X-model-score"] = str(prediction)
    logger.info('Прогнозирование выполнено успешно')
    return Rating(quality=prediction)
