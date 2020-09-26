import numpy as np
import datetime
from typing import List, Generator, Iterable, Union
import math


def validate_day(date: datetime.datetime, whitelist_days: Iterable[datetime.datetime] = None):
    """
    returns true on a weekend day (non-weekday) else returns false
    30.06.2012 is hardcoded as whitelisted date
    :param whitelist_days:
    :type date: datetime.datetime
    :param date:
    :return: bool
    """

    day_of_week = date.weekday()

    # senior condition hence the prior evaluation
    if whitelist_days:
        if date in whitelist_days:
            return True

    if day_of_week == 5 or day_of_week == 6:
        return False

    return True


def validate_eoy(date: datetime.datetime, whitelist_days: Iterable[datetime.datetime] = None):
    """
    allows only eoy months (December is approved, other months will be omitted)
    :param whitelist_days:
    :type date: datetime.datetime
    :param date:
    :return: bool
    """

    # senior condition hence the prior evaluation
    if whitelist_days:
        if date in whitelist_days:
            return True

    if date.month == 12 and date.day == 31:
        return True

    return False


def try_float_iter(itr):
    """
    converts all values to floats, if ValueError or TypeError yields np.NaN
    :param itr: iterative
    :return: generator
    """
    for elem in itr:
        try:
            yield float(elem)
        except (ValueError, TypeError):
            yield np.NaN


def try_float(elem):
    """
    converts float, if ValueError or TypeError yields np.NaN
    :param elem:
    :return: float or np.NaN
    """

    try:
        return float(elem)
    except (ValueError, TypeError):
        return np.NaN


def try_date(itr):
    """
    applies ".date()" to datetime objects
    :param itr:
    :return:
    """
    for elem in itr:
        try:
            yield elem.date()
        except (ValueError, TypeError, AttributeError):
            yield datetime.date(2020, 6, 30)


def even_groups(n: int, no_groups: int, rank_output: bool = False) -> List[int]:
    """
    returns an array of lengths normal out put
    e.g. even_groups(n=5, no_groups=2) -> [3,2]
    :param rank_output: default False; if True -> function returns an array of ranks not of group sizes
    e.g. even_groups(n=5, no_groups=2, rank_output=True) -> [1,1,1,2,2] instead of [3,2]
    :param n: total number of observations
    :param no_groups: expected amount of subgroups
    :return: List
    """

    output = []
    excess = n % no_groups

    for i in range(no_groups):
        output.append(math.floor(n / no_groups))

    for i in range(excess):
        output[i] += 1

    if not rank_output:
        return output
    else:
        rank_list = []
        for index in range(len(output)):
            for i in range(output[index]):
                rank_list.append(index+1)
        return rank_list


def annually_to_monthly(value: float):
    return (1+value)**(1/12)-1
