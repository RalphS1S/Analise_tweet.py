import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

#Ler arquivo de dados e conta a quantidade de linhas
dataset = pd.read_csv(r'C:\Users\ralph\OneDrive\Área de Trabalho\Exercícios Big Data\Tweets_Mg.csv',encoding='utf-8')
print('='*50)
print(dataset.count())
print('='*50)
print(dataset[dataset.Classificacao=='Neutro'].count())
print('='*50)
print(dataset[dataset.Classificacao=='Positivo'].count())
print('='*50)
print(dataset[dataset.Classificacao=='Negativo'].count())
print('='*50)

#Pre-Processamento
def PreprocessamentoSemStopWords(instancia):
    #remove links dos tweets
    #remove stopwords
    instancia = re.sub(r"http\S+", "", instancia).lower().replace(',','').replace('.','').replace(';','').replace('-','')
    stopwords = set(nltk.corpus.stopwords.words)('portuguese')
    palavras = [i for i in instancia.split() if not i in stopwords]
    return (" ".join(palavras))

def Stemming(instancia):
    stemmer = nltk.stem.RSLPStemmet()
    palavras=[]
    for w in instancia.split():
        palavras.append(stemmer.stem(w))
    return (" ".join(palavras))

def Preprocessamento(instancia):
    #remove links, pontos, virgulas, ponto e virgula dos tweets
    #coloca tudo em minusculo
    instancia = re.sub(r"http\S+", "", instacia).lower().replace(',','').replace('.','').replace(';','').replace('-','').replace(':','')
    return (instancia)

#Separando tweets e suas classes
tweets = dataset['Text'].values
classes = dataset['Classificacao'].values
#Gerando modelo
vectorizer = CountVectorizer(ngram_range=(1,2))
freq_tweets = vectorizer.fit_transform(tweets)
modelo = MultinomialNB()
modelo.fit(freq_tweets, classes)
#Testando o modelo com algumas instâncias simples
#Define instâncias de teste dentro de uma lista
testes = ['Esse governo está no início, vamos ver o que vai dar',
         'Estou muito feliz com o governo de Minas esse ano',
         'O estado de Minas Gerais decretou calamidade financvira!!!',
         'A segurança desse país está deixando a desejar',
         'O governador de Minas é do PT']

freq_testes = vectorizer.transform(testes)
#Fazendo classificação com o modelo treinando, obs: caso queira visualizar insira print
modelo.predict(freq_testes)
#Fazendo cross validation do modelo, obs: caso queira visualizar insira print
resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)
#Medindo a acurância média do modelo
metrics.accuracy_score(classes,resultados)
#Medida de validação do modelo
sentimento=['Positivo','Negativo','Neutro']
print(metrics.classification_report(classes,resultados,sentimento))
#Matriz de confusão
print(pd.crosstab(classes,resultados,rownames=['Real'], colnames=['Predito'], margins=True))


