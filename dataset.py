#%% Libraries
import pandas as pd
import numpy as np
import random as r
import matplotlib.pyplot as plt
from faker import Faker
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



#%% Global Variables
courseList = ['Python: Introduction','Javascript','Trigonometry','Inverse Trignonometry','Django','Applications Of Trignonometry','HTML,CSS','Node.js','Machine Learning','Data Science','Public Speaking','Literature','Grammar','Political Science']

courseDict={'frontEndList':['HTML,CSS','Javascript','React.js','jQuery','Bootstrap'],'mathsList':
            ['Trigonometry','Inverse Trignonometry','Applications Of Trignonometry'],
            'programList':['Python: Introduction','Data Science','Machine Learning','Data Structures','Algorithms'],
            'backEndList':['Node.js','Django','Ruby on Rails','pHp','Flask'],
            'politicalList':['Public Speaking','Political Science','Human Rights','Constitution'],
            'englishList':['Literature','Grammar','Vocabulary','Speech','English Expressions','Poetic Emphasis']
            }
    
typeList = ['frontEndList','mathsList','programList','backendList','politicalList','englishList']
    

fake = Faker()

#%%


def makeSingle(l):
    li=[]
    for x in l:
        for a in x:
            li.append(a)
    return li

#%%

df=pd.DataFrame()
courseColSuper,fieldColSuper = [],[]
dataDict={}

def retTimeList(n):
    return np.random.randint(10,30,n+1)

def makeADict(li):
    for x in li:
        if x in dataDict.keys():
            dataDict[x]+=1
        else:
            dataDict[x]=1

for i in range(0,1001):
    r.shuffle(courseList)
    keyList = list(courseDict.keys())
    courseCol,fieldCol = [],[]
    for i in range(0,np.random.randint(1,3)):
        i3 = np.random.randint(0,len(courseDict.keys()))
        i1,i2 = np.random.randint(0,len(courseDict[keyList[i3]])),np.random.randint(0,len(courseDict[keyList[i3]]))
        fieldCol.append(keyList[i3])
        if i1<i2:
            courseCol.append(courseDict[keyList[i3]][i1:i2])
            makeADict(courseDict[keyList[i3]][i1:i2])
        else:
            courseCol.append(courseDict[keyList[i3]][i2:i1])
            makeADict(courseDict[keyList[i3]][i2:i1])     
            
    courseColSuper.append(courseCol)
    fieldColSuper.append(fieldCol)
    
temp=[]
for x in courseColSuper:
    temp.append(makeSingle(x))

temp2=[]
for x in fieldColSuper:
    temp2.append(list(set(x)))

fieldColSuper = temp2

courseColSuper = temp
df['ID'] = np.arange(1000,2001,1)
df['Courses'] = np.array(courseColSuper)
df['TimeSpent'] = df['Courses'].apply(lambda x : retTimeList(len(x)-1))
df['Field'] = np.array(fieldColSuper)

df.to_csv('courseDetails.csv')
#%% Top Trending List

top_trending = pd.Series(dataDict).sort_values(ascending=False)[:5]


#%% Seek adjustment

speaking = pd.DataFrame()

youngNarrativeRate = [4.3,0.6]
oldNarrativeRate = [3.6,0.5]

youngReadingRate = [5.2,0.4]
oldReadingRate = [4.2,0.5]

speakingRate = pd.DataFrame()

# Youngs belong to [21,32] , Old belongs to [66,90]
youngs = np.random.randint(21,32,500)
olds = np.random.randint(66,90,500)

youngNarrativeSyllables,oldNarrativeSyllables = [],[]
for x in range(0,500):
    youngNarrativeSyllables.append(np.random.uniform(3.2,5.3))
    oldNarrativeSyllables.append(np.random.uniform(2.7,4.6))
    
youngNarrativeSyllables = np.array(youngNarrativeSyllables)
oldNarrativeSyllables = np.array(oldNarrativeSyllables)

youngNarrativeStd = youngNarrativeSyllables.std()
youngNarrativeMean = youngNarrativeSyllables.mean()
oldNarrativeStd = oldNarrativeSyllables.std()
oldNarrativeMean = oldNarrativeSyllables.mean()


# Reading Aloud

for i in range(500):
    youngNarrativeSyllables[i]*=(youngNarrativeRate[1]/youngNarrativeStd)
    youngNarrativeSyllables[i]+=(youngNarrativeRate[0]-youngNarrativeMean)
    
    oldNarrativeSyllables[i]*=(oldNarrativeRate[1]/oldNarrativeStd)
    oldNarrativeSyllables[i]+=(oldNarrativeRate[0]-oldNarrativeMean)
    

youngReadingSyllables,oldReadingSyllables = [],[]
for x in range(0,500):
    youngReadingSyllables.append(np.random.uniform(4.4,5.8))
    oldReadingSyllables.append(np.random.uniform(3.0,5.0))
    
youngReadingSyllables = np.array(youngReadingSyllables)
oldReadingSyllables = np.array(oldReadingSyllables)

youngReadingStd = youngReadingSyllables.std()
youngReadingMean = youngReadingSyllables.mean()
oldReadingStd = oldReadingSyllables.std()
oldReadingMean = oldReadingSyllables.mean()


for i in range(500):
    youngReadingSyllables[i]*=(youngReadingRate[1]/youngReadingStd)
    youngReadingSyllables[i]+=(youngReadingRate[0]-youngReadingMean)
    
    oldReadingSyllables[i]*=(oldReadingRate[1]/oldReadingStd)
    oldReadingSyllables[i]+=(oldReadingRate[0]-oldReadingMean)
    
speaking['Age'] = np.concatenate((youngs,olds))
speaking['Narrative'] = np.concatenate((youngNarrativeSyllables,oldNarrativeSyllables))
speaking['Reading'] = np.concatenate((youngReadingSyllables,oldReadingSyllables))

x_train,x_test,y_train,y_test = train_test_split(speaking['Age'].values,speaking['Narrative'].values,test_size=0.05)

regressorRead = LinearRegression()
regressorRead.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))


x_train,x_test,y_train,y_test = train_test_split(speaking['Age'].values,speaking['Narrative'].values,test_size=0.05)
regressorNarrate = LinearRegression()
regressorNarrate.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))

print("The prediction is",regressorRead.predict(np.array([44]).reshape(-1,1))[0][0],"and",regressorNarrate.predict(np.array([44]).reshape(-1,1))[0][0])




#%% Workshop dataframe


workshops = pd.DataFrame()
# workshops['Date']=np.array([datetime.datetime(2021,4,21,12,0,0),datetime.datetime(2021,4,20,2,0,0),datetime.datetime(2021,5,20,2,0,0)])
workshop_date,workshop_time = [],[]
selection=[]

for x in range(0,20):
    date_time = fake.date_time_between(start_date='now',end_date=datetime(2021,6,7,18,0,0))
    workshop_date.append(date_time.date())
    workshop_time.append(str(date_time.time().hour)+":00")
    selection.append(typeList[np.random.randint(0,len(typeList))].split("List")[0].capitalize())
    
workshops['Date'] = workshop_date
workshops['Time'] = workshop_time
workshops['Duration'] = np.array([str(x)+" hour(s)" for x in np.random.randint(1,4,20)])
workshops['Field'] = selection

workshops.to_csv('workshops.csv')







