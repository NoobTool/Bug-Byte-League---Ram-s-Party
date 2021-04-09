from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

speaking = pd.DataFrame()

youngNarrativeRate = [4.3,0.6]
oldNarrativeRate = [3.6,0.5]

youngReadingRate = [5.2,0.4]
oldReadingRate = [4.2,0.5]

youngNarrativePauseFreq=[19.4,4.7]
oldNarrativePauseFreq=[23.3,5.1]

youngReadingPauseFreq=[14.4,2.7]
oldReadingPauseFreq=[19.4,5.8]

youngNarrativePauseDur=[0.689,0.186]
oldNarrativePauseDur=[0.661,0.158]

youngReadingPauseDur=[0.551,0.137]
oldReadingPauseDur=[0.541,0.112]


speakingRate = pd.DataFrame()

#%%Youngs belong to [21,32] , Old belongs to [66,90]

youngs = np.random.randint(21,32,500)
olds = np.random.randint(66,90,500)

youngNarrativeSyllables,oldNarrativeSyllables = [],[]
youngReadingSyllables,oldReadingSyllables = [],[]

for x in range(0,500):
    youngNarrativeSyllables.append(np.random.uniform(3.2,5.3))
    oldNarrativeSyllables.append(np.random.uniform(2.7,4.6))
    
youngNarrativeSyllables = np.array(youngNarrativeSyllables)
oldNarrativeSyllables = np.array(oldNarrativeSyllables)

youngNarrativeStd = youngNarrativeSyllables.std()
youngNarrativeMean = youngNarrativeSyllables.mean()
oldNarrativeStd = oldNarrativeSyllables.std()
oldNarrativeMean = oldNarrativeSyllables.mean()


for i in range(500):
    youngNarrativeSyllables[i]*=(youngNarrativeRate[1]/youngNarrativeStd)
    youngNarrativeSyllables[i]+=(youngNarrativeRate[0]-youngNarrativeMean)
    
    oldNarrativeSyllables[i]*=(oldNarrativeRate[1]/oldNarrativeStd)
    oldNarrativeSyllables[i]+=(oldNarrativeRate[0]-oldNarrativeMean)
    


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


#%% Number of pauses

youngNarrativePauses,oldNarrativePauses = [],[]
youngNarrativePauseDurations,oldNarrativePauseDurations = [],[]
youngReadingPauses,oldReadingPauses = [],[]
youngReadingPauseDurations,oldReadingPauseDurations = [],[]

for i in range(500):
    youngNarrativePauses.append(np.random.uniform(13.0,30.2))
    oldNarrativePauses.append(np.random.uniform(13.6,31.5))
    youngReadingPauses.append(np.random.uniform(9.8,23.5))
    oldReadingPauses.append(np.random.uniform(11.4,32.4))
    
youngNarrativePauses = np.array(youngNarrativePauses)
oldNarrativePauses = np.array(oldNarrativePauses)
youngReadingPauses = np.array(youngReadingPauses)
oldReadingPauses = np.array(oldReadingPauses)
    
youngNarrativePauseStd = youngNarrativePauses.std()
youngNarrativePauseMean = youngNarrativePauses.mean()
oldNarrativePauseStd = oldNarrativePauses.std()
oldNarrativePauseMean = oldNarrativePauses.mean()

youngReadingPauseStd = youngReadingPauses.std()
youngReadingPauseMean = youngReadingPauses.mean()
oldReadingPauseStd = oldReadingPauses.std()
oldReadingPauseMean = oldReadingPauses.mean()

for i in range(500):
    youngNarrativePauses[i]*=(youngNarrativePauseFreq[1]/youngNarrativePauseStd)
    youngNarrativePauses[i]+=(youngNarrativePauseFreq[0]-youngNarrativePauseMean)
    
    oldNarrativePauses[i]*=(oldNarrativePauseFreq[1]/oldNarrativePauseStd)
    oldNarrativePauses[i]+=(oldNarrativePauseFreq[0]-oldNarrativePauseMean)
    
    youngReadingPauses[i]*=(youngReadingPauseFreq[1]/youngReadingPauseStd)
    youngReadingPauses[i]+=(youngReadingPauseFreq[0]-youngReadingPauseMean)
    
    oldReadingPauses[i]*=(oldReadingPauseFreq[1]/oldReadingPauseStd)
    oldReadingPauses[i]+=(oldReadingPauseFreq[0]-oldReadingPauseMean)
    

speaking['PauseFrequencyNarrative'] = np.concatenate((youngNarrativePauses,oldNarrativePauses))
speaking['PauseFrequencyReading'] = np.concatenate((youngReadingPauses,oldReadingPauses))

for i in range(500):
    youngReadingPauseDurations.append(np.random.uniform(0.341,0.888))
    oldReadingPauseDurations.append(np.random.uniform(0.384,0.773))
    youngNarrativePauseDurations.append(np.random.uniform(0.485,1.164))
    oldNarrativePauseDurations.append(np.random.uniform(0.468,1.124))

    
youngNarrativePauseDurations = np.array(youngNarrativePauseDurations)
oldNarrativePauseDurations = np.array(oldNarrativePauseDurations)
youngReadingPauseDurations = np.array(youngReadingPauseDurations)
oldReadingPauseDurations = np.array(oldReadingPauseDurations)
    
youngNarrativePauseDurationsStd = youngNarrativePauseDurations.std()
youngNarrativePauseDurationsMean = youngNarrativePauseDurations.mean()
oldNarrativePauseDurationsStd = oldNarrativePauseDurations.std()
oldNarrativePauseDurationsMean = oldNarrativePauseDurations.mean()

youngReadingPauseDurationsStd = youngReadingPauseDurations.std()
youngReadingPauseDurationsMean = youngReadingPauseDurations.mean()
oldReadingPauseDurationsStd = oldReadingPauseDurations.std()
oldReadingPauseDurationsMean = oldReadingPauseDurations.mean()

for i in range(500):
    youngNarrativePauseDurations[i]*=(youngNarrativePauseDur[1]/youngNarrativePauseDurationsStd)
    youngNarrativePauseDurations[i]+=(youngNarrativePauseDur[0]-youngNarrativePauseDurationsMean)
    
    oldNarrativePauseDurations[i]*=(oldNarrativePauseDur[1]/oldNarrativePauseDurationsStd)
    oldNarrativePauseDurations[i]+=(oldNarrativePauseDur[0]-oldNarrativePauseDurationsMean)
    
    youngReadingPauseDurations[i]*=(youngReadingPauseDur[1]/youngReadingPauseDurationsStd)
    youngReadingPauseDurations[i]+=(youngReadingPauseDur[0]-youngReadingPauseDurationsMean)
    
    oldReadingPauseDurations[i]*=(oldReadingPauseDur[1]/oldReadingPauseDurationsStd)
    oldReadingPauseDurations[i]+=(oldReadingPauseDur[0]-oldReadingPauseDurationsMean)
    

speaking['PauseDurationNarrative'] = np.concatenate((youngNarrativePauseDurations,oldNarrativePauseDurations))
speaking['PauseDurationReading'] = np.concatenate((youngReadingPauseDurations,oldReadingPauseDurations))



    
    









print("The prediction is",regressorRead.predict(np.array([44]).reshape(-1,1))[0][0],"and",regressorNarrate.predict(np.array([44]).reshape(-1,1))[0][0])
