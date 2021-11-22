# -*- coding: utf-8 -*-

# Load library
import pandas as pd
from datetime import datetime
import numpy as np
# Load dataset from a specific path (Note: if there is no # header, use the parameter header = None)
dataframe = pd.read_csv("MedicalAppointmentNoShows.csv")


############# Randevu Oluşturma Zamanını ve Randevu tarihini  Gün Ay Yıl özelliklerine ayırma ve bunları ana datadan çıkarma

arr = []
for i in dataframe['ScheduledDay']:
    dt = datetime.strptime(i, '%Y-%m-%dT%H:%M:%S%z')
    # Day Month Year
    m = str(dt.strftime('%d-%m-%Y')).split("-")
    arr.append(m)

df = pd.DataFrame(arr, columns = ['ScheduleDay', 'ScheduleMonth','ScheduleYear'])
dataframe = dataframe.drop("ScheduledDay", axis=1) # silme işlemi
dataframe = pd.concat([dataframe, df], axis=1, join="inner")

arr = []
for i in dataframe['AppointmentDay']:
    dt = datetime.strptime(i, '%Y-%m-%dT%H:%M:%S%z')
    # Day Month Year
    m = str(dt.strftime('%d-%m-%Y')).split("-")
    arr.append(m)

df = pd.DataFrame(arr, columns = ['AppointmentDay', 'AppointmentMonth','AppointmentYear'])
dataframe = dataframe.drop("AppointmentDay", axis=1) # silme işlemi
dataframe = pd.concat([dataframe, df], axis=1, join="inner")


dataframe.to_csv('out.csv') # düzenlenen verinin kaydedilmesi











"""
#result =  dateTimeCreator( dataframe['AppointmentDay'], ['AppointmentDay', 'AppointmentMonth','AppointmentYear'], result)
### Yaş ile ilgili istatistikler
statics1 = dataframe.describe()
statics = dataframe['Age'].describe()
### Kayıp veriler
dataframe[dataframe['Age'].isnull()]
"""