from flask import Flask,render_template
import speaking as s
import vidToAud as v
import dataset as d
import rec as r

txt = v.vidToAud('pythonIntro.mp4','pythonIntro')
videosContent,noPunc = v.theProcessing('Ya',txt)
exampleIndices,concludeIndices = v.timeStampCheck(noPunc)
data = s.predictions(5,exampleIndices,concludeIndices)
data={'seekpos1':[round(x,2) for x in data.tolist()]}

data['topList'] = d.retTopList()

app = Flask(__name__,static_url_path='/static')

r.retSim()

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html',data=data)



if __name__=="__main__":
    app.run(host="127.0.0.1")