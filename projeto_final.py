import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sklearn
import datetime
from matplotlib import *
import seaborn as sns
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import requests
from IPython.core.display import display, HTML
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import jsonify
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import *
import statsmodels

diretorio = os.path.dirname(__file__)


# Importação do Dataset
df = pd.read_csv(os.path.join(diretorio, 'TMDB_10000_Popular_Movies.csv'))

# Tratamento de dados
df['Release_Date'] = pd.to_datetime(df['Release_Date'])
df['Release_Date'] = df['Release_Date'].dt.year
df = df.dropna(subset=['Release_Date'])
df = df.astype({'Release_Date': int})
df['Profit'] = df['Revenue'] - df['Budget']  # Criação de coluna "Profit", com o Lucro de cada filme
df1 = df[df.Budget != 0]  # Remoção de filmes que constavam com orçamento 0

df['similaridade'] = 0  # Criação de coluna "similaridade", que será usada para identificar filmes parecidos

# Tratamento de dados
df = df.astype({'Title': str})
df = df.astype({'Overview': str})
df = df.astype({'Genres': str})

stopwords = nltk.corpus.stopwords.words('english')


# Função que une as colunas de título, resumo e gêneros, criando uma nova coluna 'doc'
def processa(row):
    txt = row['Title'] + ' ' + row['Overview'] + ' ' + row['Genres']

    return ' '.join([t for t in word_tokenize(txt.lower()) if (t not in stopwords) and (t not in punctuation)])


df['doc'] = df.apply(processa, axis=1)
# print(df1.shape)

# Vetorização da coluna 'doc'
vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 1),
    max_features=None,
    binary=False,
    use_idf=True
)
tfidf_matrix = vectorizer.fit_transform(df['doc'])


# Criação de 2 novos datasets separados em "Filmes de Drama" e "Filmes não-drama"
df_drama = df.loc[df['Genres'].str.contains('Drama')].copy()
df_drama['target'] = 1

df_notdrama = df.loc[~df['Genres'].str.contains('Drama')].copy()
df_notdrama['target'] = 0
df2 = df_drama
# print(df.shape)
df2 = df2.append(df_notdrama, ignore_index=True)
# print(df.shape)
df2 = df2.drop(['similaridade'], axis=1)
tfidf_matrix2 = vectorizer.fit_transform(df2['doc'])

# Separação dos dados para treinamento de modelo de Machine Learning
x = tfidf_matrix2
y = df2['target']

from sklearn.model_selection import train_test_split
from sklearn import metrics

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
)

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

std = StandardScaler(with_mean=False)
std.fit(x_train)

x_train = std.transform(x_train)
x_test = std.transform(x_test)


# Aplicação de modelo de Rede Neural
model = MLPClassifier(
    hidden_layer_sizes=(50, 15),
    activation='tanh',
    max_iter=500,
    random_state=1
)

model.fit(
    x_train, y_train
)

y_pred = model.predict(x_test)
# print(y_pred)
# print(metrics.f1_score(
#     y_test,
#     y_pred,
#     average=None  # Para múltiplas classes
# ))


def analise():
    #print('Filme mais antigo:')
    antigo = df.loc[df['Release_Date']==1902, 'Title']
    #print()
    #print('Filmes ainda não lançados:')
    futuro = df.loc[df['Release_Date']>2020, 'Title']
    return antigo, futuro


# Função que analisa o Dataset e retorna gráficos
def graficos():
    top_profit = df1.sort_values(by='Profit', ascending=False)
    top_revenue = df1.sort_values(by='Revenue', ascending=False)
    top_popularity = df1.sort_values(by='Popularity', ascending=False)
    top_rated = df1.sort_values(by='Rating_average', ascending=False)
    sns.set()
    filme_ano = df.groupby('Release_Date').count()['Title']
    plt.yticks((np.arange(0, 700, 50.0)))
    plt.ylabel('Count', fontsize=14)
    ax = filme_ano.plot(kind='bar', figsize=(21, 10), grid=True)  # Gráfico de Filmes lançados por ano
    ax.set_xlabel("Release Year", fontsize=14)
    ax.set_title('Movies Released per Year', fontsize=14)
    #plt.savefig('static/plots/movies_year.png', bbox_inches='tight')

    media = top_profit['Profit'][:20].mean()
    media_total = top_profit['Profit'].mean()
    x = top_profit['Title'][:20]
    y = top_profit['Profit'][:20]
    plt.figure(figsize=(16, 9))
    plt.axhline(media, color='r', linestyle='--')
    plt.axhline(media_total, color='black', linestyle='--')
    plt.legend(['Top 20 Mean', 'Total Mean'])
    plt.xticks(rotation='vertical')
    plt.title('Top 20 Most Profitable Movies')
    plt.ylabel('Profit (Dollars)')
    plt.bar(x, y)  # Gráfico do Top 20 Filmes mais Lucrativos
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    #plt.savefig('static/plots/profit.png', bbox_inches='tight')
    #plt.show()

    media = top_revenue['Revenue'][:20].mean()
    media_total = top_revenue['Revenue'].mean()
    x = top_revenue['Title'][:20]
    y = top_revenue['Revenue'][:20]
    plt.figure(figsize=(16, 9))
    plt.axhline(media, color='r', linestyle='--')
    plt.axhline(media_total, color='black', linestyle='--')
    plt.legend(['Top 20 Mean', 'Total Mean'])
    plt.xticks(rotation='vertical')
    plt.title('Top 20 Box Office Gross')
    plt.ylabel('Box Office Gross (Dollars)')
    plt.bar(x, y)  # Gráfico do Top 20 Filmes com maior Bilheteria
    #plt.savefig('static/plots/revenue.png', bbox_inches='tight')
    #plt.show()

    notas = np.sort(df['Rating_average'])
    plt.figure(figsize=(16, 9))
    sns.distplot(notas, hist=False, kde_kws={'cumulative': True})  # Gráfico com a distribuição das notas
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('P(rating <= x)')
    #plt.savefig('static/plots/rating_distribution.png', bbox_inches='tight')
    #plt.show()

    media = top_rated['Rating_average'][:20].mean()
    media_total = top_rated['Rating_average'].mean()
    x = top_rated['Title'][:20]
    y = top_rated['Rating_average'][:20]
    plt.figure(figsize=(16, 9))
    plt.axhline(media, color='r', linestyle='--')
    plt.axhline(media_total, color='black', linestyle='--')
    plt.legend(['Top 20 Mean', 'Total Mean'])
    plt.xticks(rotation='vertical')
    plt.title('Top 20 Best Rated Movies')
    plt.ylabel('Rating (out of 10)')
    plt.yticks((np.arange(0, 10 + 1, 0.5)))
    plt.bar(x, y)  # Gráfico do Top 20 Filmes com melhores Notas
    #plt.savefig('static/plots/rating.png', bbox_inches='tight')
    #plt.show()

    df2 = df.drop(['TMDb_Id'], axis=1)
    corr_mat = df2.corr()  # Matriz de Correlação dos dados do Dataset

    f, ax = plt.subplots(figsize=(10, 10))

    sns.heatmap(
        corr_mat,
        annot=True,
        square=True,
        vmax=1,
        vmin=-1
    )
    #plt.savefig('static/plots/correlation.png', bbox_inches='tight')

    x = top_popularity['Title'][:20]
    y = top_popularity['Popularity'][:20]
    plt.figure(figsize=(16, 9))
    plt.xticks(rotation='vertical')
    plt.title('Top 20 Movies by Popularity (in March 2020)')
    textstr = 'Popularity is a metric based on number of votes for the day, number of views for the day,\nnumber of users who marked it as a "favourite" for the day and more.\nThe higher this number, the more popular the movie is at given date.'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.8, 1.3, textstr, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    plt.ylabel('Popularity')
    plt.bar(x, y)  # Gráfico do Top 20 Filmes mais Populares em Março de 2020
    #plt.savefig('static/plots/popularity.png')
    #plt.show()

    return plt.show()


# print(df['doc'].head())


# print()
# print(vectorizer.get_feature_names()[:500])


# Função que pedia ao usuário o filme desejado e retorna um JSON com alguns dados desse filme
# Esta função foi modificada para aceitar o input diretamente da página web criada com o Flask
def busca_filme(nomefilme):
    # filme = input('Escreva um filme: ')
    filme_encoded = urllib.parse.quote(nomefilme)
    req = requests.get('https://api.themoviedb.org/3/search/movie?api_key=ef3f596bf0d222c443d5f40e548d80e0&language=en-US&query='+filme_encoded+'&page=1&include_adult=false').json()
    #dados = busca_filme(filme)
    # nome = dados['results'][0]['title']
    # poster = dados['results'][0]['poster_path']
    # poster_filme = display(HTML('<img src=https://image.tmdb.org/t/p/original/' + poster + ' style="width: 20%">'))
    # dados_filme.plot = dados['results'][0]['overview']
    # dados_filme.genre_ids = dados['results'][0]['genre_ids']
    return req


# Função que pega informações do filme buscado e faz uma análise textual para identificar 5 filmes similares
# As informações usadas para definir a similaridade foram: Nome do filme, resumo do filme e gêneros
def dados_filme(nomefilme):
    #nomefilme = input('Escreva um filme: ')
    dados = busca_filme(nomefilme)
    nome = dados['results'][0]['title']
    poster = dados['results'][0]['poster_path']
    #poster_filme = display(HTML('<img src=https://image.tmdb.org/t/p/original/' + poster + ' style="width: 20%">'))
    dados_filme.plot = dados['results'][0]['overview']
    dados_filme.genre_ids = dados['results'][0]['genre_ids']
    # runtime = int(df.loc[df['Title'] == nome]['Runtime'])
    # return nome, poster_filme

    plot = dados_filme.plot
    genre_ids = dados_filme.genre_ids
    req1 = requests.get('https://api.themoviedb.org/3/genre/movie/list?api_key=ef3f596bf0d222c443d5f40e548d80e0&language=en-US').json()
    genero = ''

    for i in req1['genres']:
        if i['id'] in genre_ids:
            genero = (genero + ' ' + i['name']).lower()
    genero = genero.lstrip()
    # print(genero)

    texto = nome + ' ' + plot + ' ' + genero
    #print(texto)
    processado = ' '.join([t for t in word_tokenize(texto.lower()) if (t not in stopwords) and (t not in punctuation)])
    #print()
    #print(processado)
    vet = vectorizer.transform([processado])
    sim = cosine_similarity(tfidf_matrix, vet)
    df['similaridade'] = sim
    similar = df.sort_values(by='similaridade', ascending=False)
    lista = []


    # print('5 filmes semelhantes:\n')
    for titulo in similar['Title'][1:6]:
        # lista.append('title:')
        lista.append(titulo)
        # print(titulo)
        filme_encoded1 = urllib.parse.quote(titulo)
        req2 = requests.get(
            'https://api.themoviedb.org/3/search/movie?api_key=ef3f596bf0d222c443d5f40e548d80e0&language=en-US&query=' + filme_encoded1 + '&page=1&include_adult=false').json()
        poster1 = req2['results'][0]['poster_path']
        if poster1 is None:
            lista.append('no poster found')
            continue
        #poster_filme1 = display(HTML('<img src=https://image.tmdb.org/t/p/original/' + poster1 + ' style="width: 20%">'))
        # lista.append('poster:')
        lista.append(poster1)

    return lista


# Função que aceita um texto informado pelo usuário e usa o modelo de rede neural para
# decidir se o texto inserido se parece com um filme dramático ou não dramático
# Esta função foi modificada para aceitar o input diretamente da página web criada com o Flask
def judge_your_script(roteiro):

    # roteiro = input('Write your script: ')
    target = model.predict(vectorizer.transform([roteiro]))[0]

    if target == 1:
        prob = model.predict_proba(vectorizer.transform([roteiro]))[:, 1][0] * 100
        response = f'Your script looks like a dramatic movie, with a chance of {prob:.2f}%'

    else:
        prob = (1-(model.predict_proba(vectorizer.transform([roteiro]))[:, 1][0])) * 100
        response = f'Your script looks like a non-dramatic movie, with a chance of {prob:.2f}%'

    return response
# # Teste com resumo de um filme dramático
#     # print(model.predict(vectorizer.transform(['Forever alone in a crowd, failed comedian Arthur Fleck seeks connection as he walks the streets of Gotham City. Arthur wears two masks -- the one he paints for his day job as a clown, and the guise he projects in a futile attempt to feel like he is part of the world around him. Isolated, bullied and disregarded by society, Fleck begins a slow descent into madness as he transforms into the criminal mastermind known as the Joker.'])))
#     #     # print(model.predict_proba(vectorizer.transform(['Forever alone in a crowd, failed comedian Arthur Fleck seeks connection as he walks the streets of Gotham City. Arthur wears two masks -- the one he paints for his day job as a clown, and the guise he projects in a futile attempt to feel like he is part of the world around him. Isolated, bullied and disregarded by society, Fleck begins a slow descent into madness as he transforms into the criminal mastermind known as the Joker.'])))
#     #     #
#     #     # # Teste com resumo de um filme não-dramático
#     #     # print(model.predict(vectorizer.transform(['Doug and his three best men go to Las Vegas to celebrate his bachelor party. However, the three best men wake up the next day with a hangover and realise that Doug is missing.'])))
#     #     # print(model.predict_proba(vectorizer.t