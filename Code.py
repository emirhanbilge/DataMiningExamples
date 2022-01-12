# -*- coding: utf-8 -*-


# Load library
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import calendar
from datetime import date


def day_of_week(my_date):
    return calendar.day_name[my_date.weekday()] 
#Read data from csv file
dataframe = pd.read_csv("MedicalAppointmentNoShows.csv")

#Splitting Appointment Creation Time and Appointment date into Day Month Year properties and extracting them from the main data
tempArray = []

#We separate the appointment date and appointment creation dates in the data as days, months and years. Required to find date ranges. 
#We will also use it to make sense of the data.
for i in dataframe['ScheduledDay']:
    dt = datetime.strptime(i, '%Y-%m-%dT%H:%M:%S%z')
    # Day Month Year
    try:
        m = str(dt.strftime('%d-%m-%Y')).split("-")
    except:
        print(m)
    tempArray.append(m)

# New Feature Names Created 
df = pd.DataFrame(tempArray, columns = ['ScheduleDay', 'ScheduleMonth','ScheduleYear'])
dataframe = dataframe.drop("ScheduledDay", axis=1) # Old feature is deleting 
dataframe = pd.concat([dataframe, df], axis=1, join="inner") # Merging of 2 dataframes



tempArray = []
dayArray = []

for i in dataframe['AppointmentDay']:
    dt = datetime.strptime(i, '%Y-%m-%dT%H:%M:%S%z')
    # Day Month Year
    m = str(dt.strftime('%d-%m-%Y')).split("-")
    dayArray.append(day_of_week(dt))
    tempArray.append(m)

df = pd.DataFrame(tempArray, columns = ['AppointmentDay', 'AppointmentMonth','AppointmentYear'])
df2 = pd.DataFrame(dayArray, columns = ['DayOfWeek'])
dataframe = dataframe.drop("AppointmentDay", axis=1) # Old feature is deleting 
#dataframe = dataframe.drop("PatientId", axis=1) # Old feature is deleting 
dataframe = dataframe.drop("AppointmentID", axis=1) # Old feature is deleting 
dataframe = pd.concat([dataframe, df], axis=1, join="inner") # Merging of 2 dataframes
dataframe = pd.concat([dataframe, df2], axis=1, join="inner") # Merging of 2 dataframes




#While testing on the data, we did the same steps again and saved it this way so that it would not lose performance 
#(We only use it for testing)
dataframe.to_csv('out.csv') # 
dataframe = pd.read_csv("out.csv" )


#Target feature must be placed last in ABT. This order has been broken as our new features have been added. Again, we take the target feature to the end.
dataframe = dataframe[["Gender","Age","Neighbourhood","Scholarship","Hipertension","Diabetes","Alcoholism","Handcap","SMS_received","ScheduleDay","ScheduleMonth","ScheduleYear","AppointmentDay","AppointmentMonth","AppointmentYear","DayOfWeek","PatientId","Noshow"]]

# Viewing the general Central Distribution properties
#print(dataframe.head())


# We create the difference between the appointment date and the appointment date as a derived feature. Later, we will examine the rate of going in those with large day ranges.
df1 = dataframe['ScheduleDay']
df2 = dataframe['AppointmentDay']
df3 = dataframe['ScheduleMonth']
df4 = dataframe['AppointmentMonth']
df5 = dataframe['ScheduleYear']
df6 = dataframe['AppointmentYear']
# Finding the difference between appointment time and get appointment time

result = (df2-df1) + (df4-df3)*30 + (df6-df5)*365
df = pd.DataFrame(result, columns = ['RangeOfTime'])

dataframe = pd.concat([dataframe, df], axis=1, join="inner")



### Finding lost data 
nullValuesAge = dataframe[dataframe['Age'].isnull()] # Nan data was not available for any of our features so we placed a few to try.
nullValuesGender= dataframe[dataframe['Gender'].isnull()]
nullValuesNeighbourhood= dataframe[dataframe['Neighbourhood'].isnull()]  # Nan data was not available for any of our features so we placed a few to try.
nullValuesScholarship = dataframe[dataframe['Scholarship'].isnull()]
nullValuesHipertension = dataframe[dataframe['Hipertension'].isnull()]
nullValuesDiabetes = dataframe[dataframe['Diabetes'].isnull()]
nullValuesAlcoholism = dataframe[dataframe['Alcoholism'].isnull()]
nullValuesHandcap= dataframe[dataframe['Handcap'].isnull()]
nullValuesSMS_received = dataframe[dataframe['SMS_received'].isnull()]
nullValuesNoShow = dataframe[dataframe['Noshow'].isnull()]


### Nan values correction
### Since most of our data is categorical, we decided to fill the missing data with that feature's mod.

dataframe['Neighbourhood'].fillna(dataframe['Neighbourhood'].mode()[0], inplace=True) # The process of finding the Mode and then filling in the values at the end of the process
nullValuesNeighbourhood= dataframe[dataframe['Neighbourhood'].isnull()] # Is the operation successful check step

dataframe['Age'].fillna(int(dataframe['Age'].mean()), inplace=True)  # The process of finding the Mode and then filling in the values at the end of the process
nullValuesAge = dataframe[dataframe['Age'].isnull()] # Is the operation successful check step

"""
********************************************************** Verilerin Temize Çekilip Kaydedilmesi
"""
dataframe = dataframe.drop("ScheduleMonth", axis=1) # Old feature is deleting 
dataframe = dataframe.drop("ScheduleDay", axis=1) # Old feature is deleting 
dataframe = dataframe.drop("ScheduleYear", axis=1) # Old feature is deleting 
dataframe = dataframe.drop("AppointmentMonth", axis=1) # Old feature is deleting 
dataframe = dataframe.drop("AppointmentDay", axis=1) # Old feature is deleting 
dataframe = dataframe.drop("AppointmentYear", axis=1) # Old feature is deleting 

dfeb = dataframe['Gender'].replace({'F':0 , 'M':1})
dfeb = dataframe['Gender'].replace({'F':0 , 'M':1})
dataframe = dataframe.drop("Gender", axis=1) # Old feature is deleting 
dataframe = pd.concat([dataframe, dfeb], axis=1, join="inner")
dfeb = dataframe['Noshow'].replace({'No':0 , 'Yes':1})
dataframe = dataframe.drop("Noshow", axis=1) # Old feature is deleting 
dataframe = pd.concat([dataframe, dfeb], axis=1, join="inner")
dataframe = dataframe.rename( {"Noshow" : "NotCame"} , axis=1)


#########################################



### Incorrect data were available in very little quantity. These were things like age being less than 0. Since we have very little erroneous data, 
### we did not go into the process of correcting them. We decided to delete them directly. 
index_names = dataframe[ dataframe['Age'] <0 ].index
dataframe.drop(index_names, inplace = True)
#checkDelete = dataframe[(dataframe['Age']==-1)]  # Checking if the deletion of faulty data was successful

### Incorrect data were available in very little quantity. These were things like age being less than 0. Since we have very little erroneous data, 
### we did not go into the process of correcting them. We decided to delete them directly. 
index_names = dataframe[ dataframe['RangeOfTime'] <0 ].index 
dataframe.drop(index_names, inplace = True)
#checkDelete = dataframe[(dataframe['RangeOfTime']<0)]   # Checking if the deletion of faulty data was successful

### Incorrect data were available in very little quantity. These were things like age being less than 0. Since we have very little erroneous data, 
### we did not go into the process of correcting them. We decided to delete them directly. 
index_names = dataframe[ dataframe['Handcap'] >1 ].index 
dataframe.drop(index_names, inplace = True)
#checkDelete = dataframe[(dataframe['Handcap'] >1)]   #Checking if the deletion of faulty data was successful





#dataframe.to_csv("Yeniveriler.csv" , index=False)


"""
# We look at the general information about the contiuounes data again after the errors are fixed.
print(dataframe.info())
staticsofData = dataframe.describe()
staticsofData =  staticsofData.drop("Diabetes", axis=1)
staticsofData =  staticsofData.drop("Hipertension", axis=1)
staticsofData =  staticsofData.drop("Alcoholism", axis=1)
staticsofData =  staticsofData.drop("Handcap", axis=1)
staticsofData =  staticsofData.drop("SMS_received", axis=1)
staticsofData =  staticsofData.drop("Scholarship", axis=1)

### Central Tandencty 
### Age related Central Tendency
staticsAge = dataframe['Age'].describe()
### Central Tendency on Time Range
staticsRangeOfTime = dataframe['RangeOfTime'].describe()
### General information about Gender
modSeriesGender = (dataframe['Gender'].value_counts())
### General information about Neighbourhood
modSeriesNeighbourhood = (dataframe['Neighbourhood'].value_counts())
### General information about Scholarship 
modSeriesScholarship = (dataframe['Scholarship'].value_counts())
### General information about Hipertension 
modSeriesHipertension = (dataframe['Hipertension'].value_counts())
### General information about Diabetes 
modSeriesDiabetes = (dataframe['Diabetes'].value_counts())
### General information about Alcoholism 
modSeriesAlcoholism = (dataframe['Alcoholism'].value_counts())
### General information about Handcap 
modSeriesHandcap = (dataframe['Handcap'].value_counts())
### General information about SMS_received 
modSeriesSMSReceived = (dataframe['SMS_received'].value_counts())
### General information about Noshow 
modSeriesNoShow = (dataframe['Noshow'].value_counts())


"""



"""
#It is a function that takes the name of feature to draw bar plots and then allows us to show how many times the feature has passed.
def bar_plot(name):
    getValueandCount = dataframe[name].value_counts().index.tolist()
    x_Val = [] #The value of the feature
    y_val=[] # How many times the value of the feature is passed
    for i in getValueandCount : 
        m = len(dataframe[(dataframe[name]==i)])
        y_val.append(m)
        x_Val.append(i)
    df = pd.DataFrame({name:x_Val, 'Size':y_val})
    ax = df.plot.bar(x=name, y='Size', rot=1,  title=name+' Bar Plot',figsize=(17,8))
    plt.xticks(rotation=90)
    #Save bar plot 
    #plt.savefig((name+".png"), dpi=300)

# Visualization of categorical data
bar_plot('Gender')
bar_plot('Neighbourhood')
#bar_plot('Scholarship')
#bar_plot('Hipertension')
#bar_plot('Diabetes')
#bar_plot('Alcoholism')
#bar_plot('Handcap')
#bar_plot('SMS_received')
bar_plot('Noshow')

#Box plot for age data
fig = plt.figure(figsize =(10, 7))
data = dataframe['Age'].to_numpy()

plt.boxplot(data)
plt.title('Age Box Plot\n',fontweight ="bold")
plt.show()


#Histogram for age data
ax = dataframe["Age"].plot(kind='hist')
dataframe["Age"].plot(kind='kde', ax=ax, secondary_y=True)
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.title('Age Histogram\n',fontweight ="bold")
plt.show()


#Box plot for RangeOfTime data
fig = plt.figure(figsize =(10, 7))
data = dataframe['RangeOfTime'].to_numpy()

plt.boxplot(data)
plt.title('RangeOfTime Box Plot\n',fontweight ="bold")
plt.show()


#Histogram for RangeOfTime data
ax = dataframe["RangeOfTime"].plot(kind='hist')
dataframe["RangeOfTime"].plot(kind='kde', ax=ax, secondary_y=True)
plt.xlabel('RangeOfTime')
plt.ylabel('Number of Appointment')
plt.title('RangeOfTime Histogram\n',fontweight ="bold")
plt.show()



# Pie chart
def pie_chart(name):
    getValueandCount = dataframe[name].value_counts().index.tolist()
    x_Val = []
    y_val=[]
    counter=0
    for i in getValueandCount: 
       m = len(dataframe[(dataframe[name]==i)])
       y_val.append(m)
       x_Val.append(i)
    df = pd.DataFrame({name:x_Val, 'Size':y_val})
    var1=y_val[0]
    var2=y_val[1]
    sizes = [(var1/(var2+var1))*100,(var2/(var2+var1))*100]
    labels = x_Val[0],x_Val[1]
    explode = (0, 0.1)  
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  
    plt.title(name+" Pie Chart")
    plt.show()

pie_chart("Gender")   
#pie_chart("Scholarship")   
#pie_chart("Hipertension")
#pie_chart("Diabetes")   
#pie_chart("Alcoholism")
#pie_chart("Handcap")
pie_chart("SMS_received")
pie_chart("Noshow")


# Combining the number of target feature according to the sent parameter
def count_group(group):
    x = dataframe.groupby([group,'Noshow'])['Noshow'].count()
    return x

# Visualization of relationships between two features
presence = dataframe.query('Noshow == "Yes"') # representation of those who did not attend the appointment
absence = dataframe.query('Noshow == "No"') # representation of those who did  attend the appointment
presence.Age.hist(alpha=0.5, bins=int(max(dataframe.Age)), label='presence', color='red')
absence.Age.hist(alpha=0.5, bins=int(max(dataframe.Age)), label='absence', color='black')
plt.title('Age Distribution')
plt.xlabel('Count People')
plt.xlabel('Age')
plt.legend()
plt.show()

# Visualization of relationships between two features
count_gender = count_group('Gender')
count_gender.plot(kind='bar',title='Counts by Gender', color=['black', 'black', 'orange', 'orange'], alpha=.7)
plt.xlabel('Gender and Noshow', fontsize=18)
plt.ylabel('Count People', fontsize=18);
plt.show()

# Visualization of relationships between two features
count_gender = count_group('SMS_received')
count_gender.plot(kind='bar',title='Counts by SMS_received', color=['black', 'black', 'orange', 'orange'], alpha=.7)
plt.xlabel('SMS_received and Noshow', fontsize=18)
plt.ylabel('Count People', fontsize=18);
plt.show()

# Visualization of relationships between two features
pd.crosstab(dataframe['Neighbourhood'],dataframe['Noshow']).plot(kind='bar',figsize=(17,8));
plt.title('Relation Between Neighbourhood & Show-up Appointments')
plt.xlabel('Neighbourhood')
plt.ylabel('Count People')
plt.show()

"""

modDaysOfWeek = (dataframe['DayOfWeek'].value_counts())
#### Saturdey sayısı 110 bin veride sadece 39 tane olduğu için atmaya karar verdik
index_names = dataframe[ dataframe['DayOfWeek'] =="Saturday" ].index 
dataframe.drop(index_names, inplace = True)
#print (dataframe['DayOfWeek'].value_counts())
### Temizlendi 39 tane saturday
hotEncodingDayOfWeek = pd.get_dummies(dataframe['DayOfWeek'][:])
hotEncodingNeighbourhood = pd.get_dummies(dataframe['Neighbourhood'][:])
dataframe = dataframe.drop("DayOfWeek", axis=1) # Old feature is deleting 
dataframe = pd.concat([dataframe, hotEncodingDayOfWeek], axis=1, join="inner")







##3 Target feature en sona alınması
targetFeature = dataframe['NotCame']
dataframe = dataframe.drop("NotCame", axis=1) # Old feature is deleting 
dataframe = pd.concat([dataframe, targetFeature], axis=1, join="inner")
###################################################



######## Eski hali için #########
dataFrameBackup = dataframe
dataframe = dataframe.drop("PatientId", axis=1) 
##################################################


######### Neighbourhoodları one hot encoding ile yeni featureler olarak türetiyoruz
dataframe = dataframe.drop("Neighbourhood", axis=1) # Old feature is deleting 
dataframe = pd.concat([dataframe, hotEncodingNeighbourhood], axis=1, join="inner")
#print(dataframe.info())
####################################################################################



########################         KORELASYON MATRİXİ         ############################################
#corr = dataframe.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
####################################################################

"""

################## **************************************** ****************************************** !!!!!!!!!!!!!!
dataframe['Num_App_Missed'] = dataframe.groupby('PatientId')['NotCame'].apply(lambda x: x.cumsum())
dataframe =dataframe.drop("PatientId", axis=1)
######################################################################################################## !!!!!!!!!
"""








from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets, neighbors
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sn
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from yellowbrick.datasets import load_occupancy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

def plotV(model, x_train, y_train, x_test, y_test):
    classes = ["Came", "Not Came"]
    from yellowbrick.classifier import classification_report
    # Instantiate the visualizer
    visualizer = classification_report(
        
        model , x_train, y_train, x_test, y_test, classes=classes, support=True
    )
    visualizer.show()


######### Age verisinin standardize edilmesi
scaler = MinMaxScaler()
x = dataframe.drop(['NotCame'], axis=1)
y = dataframe['NotCame']
x = scaler.fit_transform(x)



#verilerin egitim ve test icin bolunmesi
## Verilerin test ve train olarak bölünmesi
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#############################################################################


### Confusion Matrixin Genel Formül hale getirilmesi 
def confusion_matrix_general(clf , x_test , y_test):
   
    plot_confusion_matrix(clf, x_test, y_test) 
    plt.show()




#################### PATIENT ID VE PUANLANMA YOKKEN ################# 
#        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!             OLD FIRST TRY           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! # 


print("1 -  Dataların ilk halleri üzerinde modelin uygulanması ve başarım oranları ")

############### KNN #################### 
print("KNN with Default Parameter ")
knn = KNeighborsClassifier()
#knn.fit(x_train,y_train)
#confusion_matrix_general(knn , x_test , y_test)
#y_pred = knn.predict(x_test)
#cm = confusion_matrix(y_test,y_pred)
#print(cm)
#plotV(knn ,x_train, y_train, x_test, y_test)
#print(classification_report(y_test, y_pred))
succes = cross_val_score(estimator= knn, X=x_train, y=y_train, cv=10)
print(succes)
print(succes.mean)

print("KNN with Change Some Parameter ")
knn = KNeighborsClassifier(n_neighbors=2)
#knn.fit(x_train,y_train)
##confusion_matrix_general(knn , x_test , y_test)
#y_pred = knn.predict(x_test)
#cm = confusion_matrix(y_test,y_pred)
#print(cm)
plotV(knn ,x_train, y_train, x_test, y_test)
#print(classification_report(y_test, y_pred))
############################################

succes = cross_val_score(estimator= knn, X=x_train, y=y_train, cv=10)
print(succes)
print(succes.mean)

############# Random Forest #############
print("Random Forest with Default Parameter ")
rfc = RandomForestClassifier(n_jobs=-1)
#rfc.fit(x_train,y_train)

#y_pred = rfc.predict(x_test)
plotV(rfc ,x_train, y_train, x_test, y_test)
confusion_matrix_general(rfc , x_test , y_test)
#cm = confusion_matrix(y_test,y_pred)
#print(cm)
#print(classification_report(y_test, y_pred))

print("Random Forest with Change Some Parameters Parameter with GridSearch ")
p = [{'n_estimators':[1,2,5,10,50,100],'criterion':['gini','entropy'], 'max_features':['auto','sqrt','log2']} ]

'''
GSCV parametreleri
estimator : sınıflandırma algoritması (neyi optimize etmek istediğimiz)
param_grid : parametreler/ denenecekler
scoring: neye göre skorlanacak : örn : accuracy
cv : kaç katlamalı olacağı
n_jobs : aynı anda çalışacak iş
'''
gs = GridSearchCV(estimator= rfc, #rfc algoritması
                  param_grid = p,
                  scoring =  'accuracy',
                  cv = 10,
                  n_jobs = -1,
                  verbose=3)

grid_search = gs.fit(x_train,y_train)
bestResult = grid_search.best_score_
bestParams = grid_search.best_params_

print(bestResult)
print(bestParams)



###############################################

####################  Decision Tree Class. #################
print("Decision Tree Classifier with Default Parameter ")
dtc = DecisionTreeClassifier()

dtc.fit(x_train,y_train)

plotV(dtc ,x_train, y_train, x_test, y_test)
confusion_matrix_general(dtc , x_test , y_test)

#y_pred = dtc.predict(x_test)

#cm = confusion_matrix(y_test,y_pred)
#print(classification_report(y_test, y_pred))
#print(cm)

print("Decision Tree with Change Some Parameters Parameter with GridSearch ")
p = [{'criterion':['gini', 'entropy'],'splitter':['best', 'random'], 'max_features':['auto','sqrt','log2']} ]

'''
GSCV parametreleri
estimator : sınıflandırma algoritması (neyi optimize etmek istediğimiz)
param_grid : parametreler/ denenecekler
scoring: neye göre skorlanacak : örn : accuracy
cv : kaç katlamalı olacağı
n_jobs : aynı anda çalışacak iş
'''
gs = GridSearchCV(estimator= dtc, #rfc algoritması
                  param_grid = p,
                  scoring =  'accuracy',
                  cv = 6 ,
                  verbose=3)

grid_search = gs.fit(x_train,y_train)
bestResult = grid_search.best_score_
bestParams = grid_search.best_params_

print(bestResult)
print(bestParams)
##########################################################


#################### Logistic Regression ################
print("Logistic Regression with Default Parameter ")
logr = LogisticRegression(random_state=0 ,solver='lbfgs', max_iter=1000)
logr.fit(x_train,y_train)
y_pred = logr.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
print(cm)
confusion_matrix_general(logr , x_test , y_test)
##################################################



######################### Naive Bayes ########################
print("Naive Bayes Regression with Default Parameter ")
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
print(cm)
confusion_matrix_general(gnb , x_test , y_test)
###############################################################



#####################################################       FEATURE SAYISI VE GELİP GELMEMESİ ARASINDAKİ İLİŞKİYİ İNCELEME   ###################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ------------------------------------------------------------------------------      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
######## Now if we look korelasyon matrix , aslında bazı parameterele ihtiyacımız olmasa bile bunları yapabileceğimizi görüyoruz. 
######## En yüksek ilişkiye sahip 3 4 değişkeni oluşturarak da sonuçlara bakalım
######## Bu birkaç örnek  için denenmiştir 


########################         KORELASYON MATRİXİ         ############################################
#corr = dataframe.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
####################################################################
# En önemli featurelerin sms_received  , rangeoftime gibi değerlerdir biz de bunlardan oluşan bir model oluşturalım ve test edelim

print("2 -  Dataların corelasyon matrixine bakılarak özel seçilerek çalıştırılması  ")

scaler = MinMaxScaler()


dataframeTemp =dataFrameBackup
dataframeTemp =dataframeTemp.drop("Wednesday", axis=1)
dataframeTemp =dataframeTemp.drop("Tuesday", axis=1)
dataframeTemp =dataframeTemp.drop("Thursday", axis=1)
dataframeTemp =dataframeTemp.drop("Monday", axis=1)
dataframeTemp =dataframeTemp.drop("Friday", axis=1)
dataframeTemp =dataframeTemp.drop("Neighbourhood", axis=1)
dataframeTemp =dataframeTemp.drop("Gender", axis=1)
dataframeTemp =dataframeTemp.drop("PatientId", axis=1)
dataframeTemp =dataframeTemp.drop("Alcoholism", axis=1)


x = dataframeTemp.drop(['NotCame'], axis=1)
y = dataframeTemp['NotCame']
x = scaler.fit_transform(x)

#verilerin egitim ve test icin bolunmesi
## Verilerin test ve train olarak bölünmesi
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

############# Random Forest #############
print("Random Forest with Default Parameter ")
rfc = RandomForestClassifier(criterion ='entropy',max_features ='auto', n_estimators= 2 ,n_jobs=-1)
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test, y_pred))


print("Random Forest with Change Some Parameters Parameter with GridSearch ")
p = [{'n_estimators':[1,2,5,10,50,100],'criterion':['gini','entropy'], 'max_features':['auto','sqrt','log2']} ]

'''
GSCV parametreleri
estimator : sınıflandırma algoritması (neyi optimize etmek istediğimiz)
param_grid : parametreler/ denenecekler
scoring: neye göre skorlanacak : örn : accuracy
cv : kaç katlamalı olacağı
n_jobs : aynı anda çalışacak iş
'''
gs = GridSearchCV(estimator= rfc, #rfc algoritması
                  param_grid = p,
                  scoring =  'accuracy',
                  cv = 10,
                  n_jobs = -1,
                  verbose=3)

grid_search = gs.fit(x_train,y_train)
bestResult = grid_search.best_score_
bestParams = grid_search.best_params_

print(bestResult)
print(bestParams)



print("Decision Tree Classifier with Default Parameter ")
dtc = DecisionTreeClassifier()

dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
print(cm)
print("Decision Tree with Change Some Parameters Parameter with GridSearch ")
p = [{'criterion':['gini', 'entropy'],'splitter':['best', 'random'], 'max_features':['auto','sqrt','log2']} ]

'''
GSCV parametreleri
estimator : sınıflandırma algoritması (neyi optimize etmek istediğimiz)
param_grid : parametreler/ denenecekler
scoring: neye göre skorlanacak : örn : accuracy
cv : kaç katlamalı olacağı
n_jobs : aynı anda çalışacak iş
'''
gs = GridSearchCV(estimator= dtc, #rfc algoritması
                  param_grid = p,
                  scoring =  'accuracy',
                  cv = 6 ,
                  verbose=3)

grid_search = gs.fit(x_train,y_train)
bestResult = grid_search.best_score_
bestParams = grid_search.best_params_

print(bestResult)
print(bestParams)
################################################################################################################################################################################




######################################            Patient ID'ye göre Hastaların scorelanması ve değerlendirilerek modellerin oluşturulması      ####################
#########  En iyi sonuçlara geçilmesi 
print("3 -  Yeni featurenin türetilmesi ve gereksiz dataların temizlenerek algoritmanın optimize edilmesi ")

### Dataframemizde Patient Id silmiştik onu tekrar dataframe ekleyelim 
dataframe = dataFrameBackup 
corr = dataframe.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
### Şimdi hastaların score hesabını yapalım                  
################## **************************************** ****************************************** !!!!!!!!!!!!!!
dataframe['Num_App_Missed'] = dataframe.groupby('PatientId')['NotCame'].apply(lambda x: x.cumsum())
dataframe =dataframe.drop("PatientId", axis=1)
######################################################################################################## !!!!!!!!!!!!


#### Neighbourhood gereksiz ve etkilemeyen bir veri olduğu için temizleyelim
dataframe =dataframe.drop("Neighbourhood", axis=1)




######### Age verisinin standardize edilmesi
scaler = MinMaxScaler()
x = dataframe.drop(['NotCame'], axis=1)
y = dataframe['NotCame']
x = scaler.fit_transform(x)



#verilerin egitim ve test icin bolunmesi
## Verilerin test ve train olarak bölünmesi
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)



############### KNN #################### 
print("KNN with Default Parameter ")
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
#confusion_matrix_general(knn , x_test , y_test)
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test, y_pred))
#plotV(knn ,x_train, y_train, x_test, y_test)
#confusion_matrix_general(knn , x_test , y_test)

print("KNN with Change Some Parameter ")
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
#confusion_matrix_general(knn , x_test , y_test)
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test, y_pred))
############################################


############# Random Forest #############
print("Random Forest with Default Parameter ")
rfc = RandomForestClassifier( n_jobs=-1)
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test, y_pred))

#plotV(rfc ,x_train, y_train, x_test, y_test)
#confusion_matrix_general(rfc , x_test , y_test)



from sklearn.model_selection import learning_curve

dtc = DecisionTreeClassifier()
train_sizes, train_scores, test_scores = learning_curve(AdaBoostClassifier(base_estimator = dtc), x, y, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace( 0.01, 1.0, 50))
# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
# Draw bands
plt.fill_between(train_sizes, train_mean - train_std,
train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std,
test_mean + test_std, color="#DDDDDD")
# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
plt.legend(loc="best")
plt.tight_layout()
plt.show()


#confusion_matrix_general(rfc , x_test , y_test)
print("Random Forest with Change Some Parameters Parameter with GridSearch ")
p = [{'n_estimators':[1,2,5,10,50,100],'criterion':['gini','entropy'], 'max_features':['auto','sqrt','log2']} ]

'''
GSCV parametreleri
estimator : sınıflandırma algoritması (neyi optimize etmek istediğimiz)
param_grid : parametreler/ denenecekler
scoring: neye göre skorlanacak : örn : accuracy
cv : kaç katlamalı olacağı
n_jobs : aynı anda çalışacak iş
'''
gs = GridSearchCV(estimator= rfc, #rfc algoritması
                  param_grid = p,
                  scoring =  'accuracy',
                  cv = 10,
                  n_jobs = -1,
                  verbose=3)

grid_search = gs.fit(x_train,y_train)
bestResult = grid_search.best_score_
bestParams = grid_search.best_params_

print(bestResult)
print(bestParams)


###############################################

####################  Decision Tree Class. #################
print("Decision Tree Classifier with Default Parameter ")
dtc = DecisionTreeClassifier()

dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
print(cm)

print("Decision Tree with Change Some Parameters Parameter with GridSearch ")
p = [{'criterion':['gini', 'entropy'],'splitter':['best', 'random'], 'max_features':['auto','sqrt','log2']} ]

'''
GSCV parametreleri
estimator : sınıflandırma algoritması (neyi optimize etmek istediğimiz)
param_grid : parametreler/ denenecekler
scoring: neye göre skorlanacak : örn : accuracy
cv : kaç katlamalı olacağı
n_jobs : aynı anda çalışacak iş
'''
gs = GridSearchCV(estimator= dtc, #rfc algoritması
                  param_grid = p,
                  scoring =  'accuracy',
                  cv = 6 ,
                  verbose=3)

grid_search = gs.fit(x_train,y_train)
bestResult = grid_search.best_score_
bestParams = grid_search.best_params_

print(bestResult)
print(bestParams)
##########################################################

#################### Logistic Regression ################
print("Logistic Regression with Default Parameter ")
logr = LogisticRegression(random_state=0 ,solver='lbfgs', max_iter=1000)
logr.fit(x_train,y_train)
y_pred = logr.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
print(cm)
confusion_matrix_general(logr , x_test , y_test)
plotV(logr ,x_train, y_train, x_test, y_test)


##################################################


######################### Naive Bayes ########################
print("Naive Bayes Regression with Default Parameter ")
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
print(cm)
confusion_matrix_general(gnb , x_test , y_test)
###############################################################



######################### AdaBoost Bayes ########################
print("Ada Boost DTC")
dtc = DecisionTreeClassifier()
ada = AdaBoostClassifier(base_estimator = dtc)
ada.fit(x_train, y_train)

y_pred = ada.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
print(cm)
confusion_matrix_general(ada , x_test , y_test)
##########################################################

