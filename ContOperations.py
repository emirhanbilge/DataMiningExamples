
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sns


# Derived Featurenin oluşturulması randevu zamanı ile gidiceği tarih arasındaki gün farkı
dataframe = pd.read_csv("out.csv" )

print(dataframe.head())


df1 = dataframe['ScheduleDay']
df2 = dataframe['AppointmentDay']
df3 = dataframe['ScheduleMonth']
df4 = dataframe['AppointmentMonth']
df5 = dataframe['ScheduleYear']
df6 = dataframe['AppointmentYear']


result = (df2-df1) + (df4-df3)*30 + (df6-df5)*365
df = pd.DataFrame(result, columns = ['RangeOfTime'])
dataframe = pd.concat([dataframe, df], axis=1, join="inner")
#dataframe.to_csv('out2.csv') # düzenlenen verinin kaydedilmesi


#result =  dateTimeCreator( dataframe['AppointmentDay'], ['AppointmentDay', 'AppointmentMonth','AppointmentYear'], result)



### Kayıp veriler
nullValuesAge = dataframe[dataframe['Age'].isnull()] # Nan verimiz  var ve bilerek yerleştirdik
nullValuesGender= dataframe[dataframe['Gender'].isnull()]
nullValuesNeighbourhood= dataframe[dataframe['Neighbourhood'].isnull()]  # Nan verimiz  var ve bilerek yerleştirdik
nullValuesScholarship = dataframe[dataframe['Scholarship'].isnull()]
nullValuesHipertension = dataframe[dataframe['Hipertension'].isnull()]
nullValuesDiabetes = dataframe[dataframe['Diabetes'].isnull()]
nullValuesAlcoholism = dataframe[dataframe['Alcoholism'].isnull()]
nullValuesHandcap= dataframe[dataframe['Handcap'].isnull()]
nullValuesSMS_received = dataframe[dataframe['SMS_received'].isnull()]
nullValuesNoShow = dataframe[dataframe['No-show'].isnull()]



### Nan valuesi düzeltme 
### Kategorik veri olduğu için o featurenin modu ile doldurma kararı aldık


dataframe['Neighbourhood'].fillna(dataframe['Neighbourhood'].mode()[0], inplace=True)
nullValuesNeighbourhood= dataframe[dataframe['Neighbourhood'].isnull()]


dataframe['Age'].fillna(int(dataframe['Age'].mean()), inplace=True)
nullValuesNeighbourhood= dataframe[dataframe['Age'].isnull()]
nullValuesAge = dataframe[dataframe['Age'].isnull()]


### Hatalı verileri silme 
index_names = dataframe[ dataframe['Age'] <0 ].index
dataframe.drop(index_names, inplace = True)
#checkDelete = dataframe[(dataframe['Age']==-1)]  

index_names = dataframe[ dataframe['RangeOfTime'] <0 ].index
dataframe.drop(index_names, inplace = True)
#checkDelete = dataframe[(dataframe['RangeOfTime']<0)]  

index_names = dataframe[ dataframe['Handcap'] >1 ].index
dataframe.drop(index_names, inplace = True)
#checkDelete = dataframe[(dataframe['Handcap'] >1)]  



# Veri hakkındaki genel bilgiler 
print(dataframe.info())

#statics1 = dataframe.describe()

### Central Tandencty 
### Yaş ile ilgili Central Tandencty 
staticsAge = dataframe['Age'].describe()
### Zaman Aralığı ile ilgili Central Tandencty 
staticsRangeOfTime = dataframe['RangeOfTime'].describe()



modSeriesGender = (dataframe['Gender'].value_counts())
modSeriesNeighbourhood = (dataframe['Neighbourhood'].value_counts())
modSeriesScholarship = (dataframe['Scholarship'].value_counts())
modSeriesHipertension = (dataframe['Hipertension'].value_counts())
modSeriesDiabetes = (dataframe['Diabetes'].value_counts())
modSeriesAlcoholism = (dataframe['Alcoholism'].value_counts())
modSeriesHandcap = (dataframe['Handcap'].value_counts())
modSeriesSMSReceived = (dataframe['SMS_received'].value_counts())
modSeriesNoShow = (dataframe['No-show'].value_counts())


def bar_plot(name):
    eb = dataframe[name].value_counts().index.tolist()
    x_Val = []
    y_val=[]
    for i in eb : 
        
        m = len(dataframe[(dataframe[name]==i)])
        y_val.append(m)
        x_Val.append(i)
    df = pd.DataFrame({name:x_Val, 'Size':y_val})
    ax = df.plot.bar(x=name, y='Size', rot=0)
# Categoric verilerin görselleştirilmesi
bar_plot('Gender')
bar_plot('Neighbourhood')
bar_plot('Scholarship')
bar_plot('Hipertension')
bar_plot('Diabetes')
bar_plot('Alcoholism')
bar_plot('Handcap')
bar_plot('SMS_received')
bar_plot('No-show')

#Box plor for age data
fig = plt.figure(figsize =(10, 7))
data = dataframe['Age'].to_numpy()

plt.boxplot(data)
plt.title('Age Box Plot\n',
          fontweight ="bold")
plt.show()

#Histogram for age data
plt.hist(data, bins=5, color ='gray')
plt.xlabel('Age')
plt.ylabel('Number of People')
  
plt.title('Age Histogram\n',
          fontweight ="bold")
plt.show()

#Box plor for RangeOfTime data
fig = plt.figure(figsize =(10, 7))
data = dataframe['RangeOfTime'].to_numpy()

plt.boxplot(data)
plt.title('RangeOfTime Box Plot\n',
          fontweight ="bold")
plt.show()

#Histogram for RangeOfTime data
plt.hist(data, bins=5, color ='gray')
plt.xlabel('RangeOfTime')
plt.ylabel('Number of People')
  
plt.title('RangeOfTime Histogram\n',
          fontweight ="bold")
plt.show()