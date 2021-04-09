#%% Library Imports

import subprocess
import time
import speech_recognition as sr
import re
import string
import nltk
import asyncio
import json
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk import ngrams
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz,process


#%% Global Variables

start = time.time()
r = sr.Recognizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.remove("for")
stop_words.remove("while")
stop_words.remove("do")
stop_words.remove("that")
d={}
videosContent=[]
examples=["instance","suppose that","imagine","pretend that"
          ,"let say","such as",]

conclusions=['conclude','recapitulate','sum up']

#%% Video Processing Functions


def audToText(fileName):
    
    with sr.AudioFile("/home/ramji/hk/{}".format(fileName.split(".")[0]+'.wav')) as source:
        audio_text = r.listen(source,phrase_time_limit=10000,timeout=5)
        try:
            text = r.recognize_google(audio_text).lower()
            
        except Exception as e:
            text = str(e)
            
        finally:
            return text
            


def vidToAud(fileName,audioname):
    cmd = 'ffmpeg -y -i "{}" \
        -ab 160k -ac 2 -ar 44100 -vn "/home/ramji/hk/{}.wav"'.format(fileName,audioname.split('.')[0])
    subprocess.call(cmd,shell=True)
    text = audToText(audioname)
    return text

#%% Natural Language Processing

def theProcessing(name,text):
    
    d,vidContent={},{}
    keywords=[]
    
    #Removing all numbers
    noNum = re.sub(r'\d+','',text)
    
    #Removing all punctuations
    noPunc = noNum.translate(str.maketrans("","", string.punctuation))
    
    #Removing stopwords
    tokens = word_tokenize(noPunc)
    noStops = [i for i in tokens if not i in stop_words]
    
    #Stemming each word
    noInflections = []
    for word in noStops:
        noInflections.append(lemmatizer.lemmatize(word))
            
    
    for word in noInflections:
        if word in d:
            d[word]+=1
        else:
            d[word]=1
            
    max_freq = max(d.values())
    
    for word in d.keys():
        d[word]=d[word]/max_freq
        if d[word]>0.33: keywords.append(word)
    
    vidContent["problem_name"] = name
    vidContent["keywords"] = keywords
    
    return vidContent,noPunc
    
def timeStampCheck(text):
    exampleIndices=[]
    concludeIndices=[]
    text+=" for instance"
    tokens = word_tokenize(text)
    token2 = ngrams(text.split(),2)
    for x in tokens:
        if x in examples:
            exampleIndices.append(tokens.index(x))
        if x in conclusions:
            concludeIndices.append(tokens.index(x))
            
    for x in token2:
        if x in examples:
            exampleIndices.append(token2.index(x))
        if x in conclusions:
            concludeIndices.append(token2.index(x))
            
    return exampleIndices,concludeIndices

    
    

#%% The Search

searchResult ={}
txt = vidToAud('pythonIntro.mp4','pythonIntro')
videosContent,noPunc = theProcessing('Ya',txt)
exampleIndices,concludeIndices = timeStampCheck(noPunc)

# for content in videosContent:
#     print(process.extract("layout",content["keywords"],limit=1,scorer=fuzz.token_set_ratio),"\n\n")
#     print(content["id"])




    