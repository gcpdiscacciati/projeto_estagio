import json
from flask import Flask
from flask import render_template
from flask import Response, request, jsonify, url_for
import os
print('Loading... This process may take a while.')
import projeto_final


app = Flask(__name__, static_url_path='/static')



@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        home.nomefilme = request.form.get('nomefilme')

    return render_template('./index.html')


@app.route('/busca')
def busca():
    nomefilme = home.nomefilme
    return jsonify(projeto_final.busca_filme(nomefilme))


@app.route('/similar')
def similar():
    nomefilme = home.nomefilme
    return jsonify(projeto_final.dados_filme(nomefilme))


@app.route('/judge-your-script', methods=['GET', 'POST'])
def writer():
    if request.method == 'POST':
        writer.roteiro = request.form.get('roteiro')

    return render_template('roteiro.html')


@app.route('/roteiros')
def roteiros():
    roteiro = writer.roteiro
    return jsonify(projeto_final.judge_your_script(roteiro))


@app.route('/plots')
def graficos():
    return render_template('graficos.html')


app.run(debug=True)
