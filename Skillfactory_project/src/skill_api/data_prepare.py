def iqr(x):
    """
    Функция для расчета межквартильного размаха.
    На входе функции df['column']
    """
    return x.quantile(0.75) - x.quantile(0.25)


def perc25(x):
    """
    Функция для расчета нижнего квартиля.
    На входе функции df['column']
    """
    return x.quantile(0.25)


def perc75(x):
    """
    Функция для расчета верхнего квартиля.
    На входе функции df['column']
    """
    return x.quantile(0.75)


def outlier_low(x):
    """
    Функция для расчета границы нижнего выброса.
    На входе функции df['column']
    """
    return perc25(x) - 1.5 * iqr(x)


def outlier_high(x):
    """
    Функция для расчета границы верхнего выброса.
    На входе функции df['column']
    """
    return perc75(x) + 1.5 * iqr(x)
