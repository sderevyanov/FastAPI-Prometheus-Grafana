import numpy as np
from xgboost import XGBRegressor
from datetime import datetime
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor


def catboost_train(x_train, x_val, y_train, y_val, model_params, logger):
    """
    Функция для переобучения модели на алгоритме CatBoostRegressor в методе forecast_with_train.

    :param x_train: сформированная выборка данных для обучения
    :param x_val: сформированная выборка данных для валидации
    :param y_train: целевая переменная для обучения
    :param y_val: целевая переменная для валидации
    :param model_params: гиперпараметры изначально обученной модели
    :param logger: заданный логгер
    :return: переобученная модель, каждый спрогнозированный период добавляется в обучающкю выборку для переобучения
            модели и прогноза следующего периода.
    """
    # Зададим время старта выполнения функции
    start_time = datetime.now()
    try:
        model = CatBoostRegressor(iterations=model_params["iterations"],
                                  random_seed=model_params["random_seed"],
                                  loss_function=model_params["loss_function"],
                                  eval_metric=model_params["eval_metric"],
                                  custom_metric=model_params["custom_metric"],
                                  silent=model_params["silent"],
                                  learning_rate=model_params["learning_rate"],
                                  l2_leaf_reg=model_params["l2_leaf_reg"],
                                  depth=model_params["depth"]
                                  )
    except:
        logger.error('Один из гиперпараметров CatBoostRegressor задан неверно.')
        logger.info('Завершено')
        error = {'Forecast_time': str(datetime.now() - start_time),
                 'Status': str('Error'),
                 'Message': str('Один из гиперпараметров CatBoostRegressor задан неверно.'),
                 'Data_predict': [dict()]}
        return error

    # Обучим модель CatBoostRegressor
    try:
        train_dataset = Pool(data=x_train, label=y_train)
        eval_dataset = Pool(data=x_val, label=y_val)
        model.fit(train_dataset,
                  eval_set=eval_dataset,
                  early_stopping_rounds=50,
                  verbose=False)
    except:
        logger.error('Возникла ошибка при обучении модели CatBoostRegressor.')
        logger.info('Завершено')
        error = {'Forecast_time': str(datetime.now() - start_time),
                 'Status': str('Error'),
                 'Message': str('Возникла ошибка при обучении модели CatBoostRegressor.'),
                 'Data_predict': [dict()]}
        return error
    return model


def randomforest_train(x_train, x_val, y_train, y_val, model_params, logger):
    """
    Функция для переобучения модели на алгоритме RandomForestRegressor в методе forecast_with_train.

    :param x_train: сформированная выборка данных для обучения
    :param x_val: сформированная выборка данных для валидации
    :param y_train: целевая переменная для обучения
    :param y_val: целевая переменная для валидации
    :param model_params: гиперпараметры изначально обученной модели
    :param logger: заданный логгер
    :return: переобученная модель, каждый спрогнозированный период добавляется в обучающкю выборку для переобучения
             модели и прогноза следующего периода.
    """
    # Зададим время старта выполнения функции
    start_time = datetime.now()
    try:
        x_train = np.vstack((x_train, x_val))
        y_train = np.vstack((y_train, y_val))
        model = RandomForestRegressor(n_estimators=model_params["n_estimators"],
                                      max_features=model_params["max_features"],
                                      max_depth=model_params["max_depth"],
                                      min_samples_split=model_params["min_samples_split"],
                                      min_samples_leaf=model_params["min_samples_leaf"],
                                      bootstrap=model_params["bootstrap"],
                                      random_state=model_params["random_state"])
    except:
        logger.error('Один из гиперпараметров RandomForestRegressor задан неверно.')
        logger.info('Завершено')
        error = {'Forecast_time': str(datetime.now() - start_time),
                 'Status': str('Error'),
                 'Message': str('Один из гиперпараметров RandomForestRegressor задан неверно.'),
                 'Data_predict': [dict()]}
        return error
    # Обучим модель RandomForestRegressor
    try:
        model.fit(x_train, y_train.ravel())
    except:
        logger.error('Возникла ошибка при обучении модели RandomForestRegressor.')
        logger.info('Завершено')
        error = {'Forecast_time': str(datetime.now() - start_time),
                 'Status': str('Error'),
                 'Message': str('Возникла ошибка при обучении модели RandomForestRegressor.'),
                 'Data_predict': [dict()]}
        return error
    return model


def extratree_train(x_train, x_val, y_train, y_val, model_params, logger):
    """
     Функция для переобучения модели на алгоритме ExtraTreesRegressor в методе forecast_with_train.

     :param x_train: сформированная выборка данных для обучения
     :param x_val: сформированная выборка данных для валидации
     :param y_train: целевая переменная для обучения
     :param y_val: целевая переменная для валидации
     :param model_params: гиперпараметры изначально обученной модели
     :param logger: заданный логгер
     :return: переобученная модель, каждый спрогнозированный период добавляется в обучающкю выборку для переобучения
              модели и прогноза следующего периода.
     """
    # Зададим время старта выполнения функции
    start_time = datetime.now()
    try:
        x_train = np.vstack((x_train, x_val))
        y_train = np.vstack((y_train, y_val))
        model = ExtraTreesRegressor(n_estimators=model_params["n_estimators"],
                                    max_features=model_params["max_features"],
                                    max_depth=model_params["max_depth"],
                                    min_samples_split=model_params["min_samples_split"],
                                    min_samples_leaf=model_params["min_samples_leaf"],
                                    bootstrap=model_params["bootstrap"],
                                    random_state=model_params["random_state"])
    except:
        logger.error('Один из гиперпараметров ExtraTreesRegressor задан неверно.')
        logger.info('Завершено')
        error = {'Forecast_time': str(datetime.now() - start_time),
                 'Status': str('Error'),
                 'Message': str('Один из гиперпараметров ExtraTreesRegressor задан неверно.'),
                 'Data_predict': [dict()]}
        return error

    # Обучим модель ExtraTreesRegressor
    try:
        model.fit(x_train, y_train.ravel())
    except:
        logger.error('Возникла ошибка при обучении модели ExtraTreesRegressor.')
        logger.info('Завершено')
        error = {'Forecast_time': str(datetime.now() - start_time),
                 'Status': str('Error'),
                 'Message': str('Возникла ошибка при обучении модели ExtraTreesRegressor.'),
                 'Data_predict': [dict()]}
        return error
    return model


def xgboost_train(x_train, x_val, y_train, y_val, model_params, logger):
    """
      Функция для переобучения модели на алгоритме XGBRegressor в методе forecast_with_train.

      :param x_train: сформированная выборка данных для обучения
      :param x_val: сформированная выборка данных для валидации
      :param y_train: целевая переменная для обучения
      :param y_val: целевая переменная для валидации
      :param model_params: гиперпараметры изначально обученной модели
      :param logger: заданный логгер
      :return: переобученная модель, каждый спрогнозированный период добавляется в обучающкю выборку для переобучения
               модели и прогноза следующего периода.
      """
    # Зададим время старта выполнения функции
    start_time = datetime.now()
    try:
        model = XGBRegressor(objective=model_params["objective"],
                             max_leaves=10,
                             learning_rate=model_params["learning_rate"],
                             max_depth=model_params["max_depth"],
                             min_child_weight=model_params["min_child_weight"],
                             n_estimators=model_params["n_estimators"]
                             )
    except:
        logger.error('Один из гиперпараметров XGBRegressor задан неверно.')
        logger.info('Завершено')
        error = {'Forecast_time': str(datetime.now() - start_time),
                 'Status': str('Error'),
                 'Message': str('Один из гиперпараметров XGBRegressor задан неверно.'),
                 'Data_predict': [dict()]}
        return error

    # Обучим модель XGBRegressor
    try:
        model.fit(x_train, y_train,
                  eval_set=[(x_train, y_train), (x_val, y_val)],
                  # early_stopping_rounds=30,
                  verbose=False)
    except:
        logger.error('Возникла ошибка при обучении модели XGBRegressor.')
        logger.info('Завершено')
        error = {'Forecast_time': str(datetime.now() - start_time),
                 'Status': str('Error'),
                 'Message': str('Возникла ошибка при обучении модели XGBRegressor.'),
                 'Data_predict': [dict()]}
        return error
    return model


def staking_train(x_train, x_val, y_train, y_val, model_params, logger):
    """
     Функция для переобучения модели на алгоритме StackingRegressor в методе forecast_with_train.

     :param x_train: сформированная выборка данных для обучения
     :param x_val: сформированная выборка данных для валидации
     :param y_train: целевая переменная для обучения
     :param y_val: целевая переменная для валидации
     :param model_params: гиперпараметры изначально обученной модели
     :param logger: заданный логгер
     :return: переобученная модель, каждый спрогнозированный период добавляется в обучающкю выборку для переобучения
              модели и прогноза следующего периода.
     """
    # Зададим время старта выполнения функции
    start_time = datetime.now()
    try:
        x_train = np.vstack((x_train, x_val))
        y_train = np.vstack((y_train, y_val))
        estimators = [
            ('ctbr', CatBoostRegressor(random_state=42, verbose=False)),
            ('etr', ExtraTreesRegressor(random_state=42)),
            ('xgb', XGBRegressor(random_state=42)),
            ('lr', LinearRegression()),
            ('knn', KNeighborsRegressor(n_neighbors=5))
        ]
        reg = StackingRegressor(estimators=estimators,
                                  final_estimator=RandomForestRegressor(n_estimators=100, random_state=42))

        model = Pipeline([('stacking', reg)])
    except:
        logger.error('Один из гиперпараметров StackingRegressor задан неверно.')
        logger.info('Завершено')
        error = {'Forecast_time': str(datetime.now() - start_time),
                 'Status': str('Error'),
                 'Message': str('Один из гиперпараметров StackingRegressor задан неверно.'),
                 'Data_predict': [dict()]}
        return error

    # Обучим модель ExtraTreesRegressor
    try:
        model.fit(x_train, y_train.ravel())
    except:
        logger.error('Возникла ошибка при обучении модели StackingRegressor.')
        logger.info('Завершено')
        error = {'Forecast_time': str(datetime.now() - start_time),
                 'Status': str('Error'),
                 'Message': str('Возникла ошибка при обучении модели StackingRegressor.'),
                 'Data_predict': [dict()]}
        return error
    return model
