#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: gjwei
import pandas as pd
import datetime


def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df
