from flask import Flask
from flask import render_template
from flask import request
from ZTJNMZKBackEnd import GetCosineSimilarity
from ZTJNMZKBackEnd import WriteToTxt

app = Flask(__name__)

@app.route('/')
	return render_template('zoekscherm.html')

@app.route('/zoekopdracht', methods=['POST'])
def resultaat():
	query = request.form["query"]
	result_list = [] #We maken hier de lijst result_list opnieuw aan, omdat er anders problemen/incosequent gedrag op kan leveren ivm de lijst die wordt meegegeven 
	result_list = GetCosineSimilarity(query)
	return render_template('resultaat.html', gezocht=query, result_list=result_list[0:5]) #Om maar 5 artikelen te laten zien, worden alleen artikel 0 t/m artikel 5 laten zien 

@app.route('/addtxt', methods=['POST'])
def addtxt():
	return render_template('addtxt.html')

@app.route('/leesbestand', methods=['POST'])
def leesbestand():
	titel = request.form['titel'] #voor meer informatie, zie resultaat.html
	text = request.form['text']
	return render_template('leesbestand.html', titel=titel, text=text)

@app.route('/addtxt2', methods=['POST'])
def addtxt2():
	titel = request.form["titel"] #De tekst en de titel die worden meegenomen naar de WriteToTxt functie komen van van de pagina addtxt.html
	text = request.form["text"]
	return render_template('addtxt2.html', show_message=WriteToTxt(titel, text))
	 

if __name__ == "__main__":
	app.config['TEMPLATES_AUTO_RELOAD']=True
	app.config['DEBUG'] = True
	app.config['SERVER_NAME'] = "127.0.0.1:6969"  #hehe nice      
	app.run()