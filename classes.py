#https://mode.com/example-gallery/python_dataframe_styling/

#bibliothèque

import plotly.express as px
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.metrics import r2_score
import math
# Fonction pour changer la couleur du texte
# def couleur(valeur):
#     if valeur < 0:
#         return f'<font color="red">{valeur}</font>'
#     else:
#         return f'<font color="green">{valeur}</font>'
    
# import os

#https://mode.com/example-gallery/python_dataframe_styling/

class portefeuille:
    def __init__(self, noms_titres, quantites, time_achat) -> None:
        self.noms_titres = noms_titres
        self.quantites = quantites
        self.time = time_achat
        #On importe les données
        self.data = []
        self.prix = []
        for i in self.noms_titres:
            path = "/Users/maximemoreau/Desktop/PEPE/data-cac40/"+i+".csv"
            df = pd.read_csv(path)
            self.data.append(df)
            #print(i, "chargé")
        #On récupère le prix de fermeture des actions à l'instant t = time_achat
        self.recuperer_prix(self.time)
        #Création du dataframe de notre portefeuille
        self.complete_actions_df()
    

    def __str__(self) -> str:
        return f'{self.actions}'
    
    def get_noms_titres(self):
        return self.noms_titres

    def recuperer_prix(self, t):
        for i in self.data:
            self.prix.append(i['Close'][t])

            
    def get_data(self):
        return self.data
    
    def complete_actions_df(self):
        self.actions_temp = {'Valeur à t = ' + str(self.time) :self.prix,
                        'Quantite': self.quantites}
        self.actions = pd.DataFrame(self.actions_temp, index=self.noms_titres)
        self.actions['Total en €'] = self.actions.apply(lambda row: row['Valeur à t = ' + str(self.time)] * row['Quantite'], axis=1)

    def add_predic(self, t, mod):
            predict = []
            for i in mod.models:
                predict.append(i.predict(np.array([t]).reshape(1,1))[0])
            self.actions['Valeur estimée à t = '+str(t)] = predict
            self.actions['Variation unitaire estimée en %'] = 100*(predict-self.actions['Valeur à t = ' + str(self.time)])/self.actions['Valeur à t = ' + str(self.time)]
            self.actions['Total estimé à = '+str(t)] = self.actions.apply(lambda row: row['Valeur estimée à t = '+str(t)] * row['Quantite'], axis=1)

            #self.couleur_evolution()
            self.actions['Coefficient de determination'] = self.scores
            self.save = self.actions
            # self.actions['Variance entre t = 0 et t = ' + str(self.time)] =
    def afficher_resultats(self):
        return self.actions.head(len(self.actions))
    
    def couleur_evolution(self):
        self.actions['Variation unitaire estimée en %'] = self.actions['Variation unitaire estimée en %'].apply(lambda x: couleur(x))

    def get_r2_mean(self):
        return self.actions['Coefficient de determination'].mean()
    
    def get_r2_var(self):
        return self.actions['Coefficient de determination'].var()

    def set_scores(self, scores):
        self.scores = scores

    def df_update(self):
        self.actions.iloc[:,2] = self.actions.iloc[:,0]*self.actions.iloc[:,1]
        self.actions.iloc[:,5] = self.actions.iloc[:,3]*self.actions.iloc[:,1]
        self.actions.iloc[:,4] = 100*(self.actions.iloc[:,3]-self.actions.iloc[:,0])/self.actions.iloc[:,0]

    def trier_portefeuil(self, variation_min, var_max, r2_min):
        self.actions = self.save
        self.actions = self.actions.drop(self.actions[(self.actions.iloc[:,4] < variation_min) | (self.actions.iloc[:,-1] > var_max) | (self.actions.iloc[:,-2] < r2_min)].index)

    def variance_p(self):
        poids = []
        q = self.actions.iloc[:,0].sum()
        for i in self.actions.iloc[:,0] :
            poids.append(i/q)
        v = 0
        index = self.actions.index.tolist()
        for i in range(len(index)):
            for j in range(len(index)):
                k = index[i]
                l = index[j]
                a = pd.read_csv("data-cac40/" +k+".csv").head(self.time)['Close'].to_numpy()
                b = pd.read_csv('data-cac40/'+l+".csv").head(self.time)['Close'].to_numpy()
                v += np.cov(a,b)[0][1]*poids[i]*poids[j]
        return v
            
class lr_models:
    #period = période d'apprentissage => inputs
    def __init__(self, portefeuille, period, learning_rate) -> None:
        self.mon_portefeuille = portefeuille 
        self.noms_titres = portefeuille.get_noms_titres()
        self.m = period
        self.learning_rate = learning_rate
        self.scores = []
        self.vars = []
    def create_models(self):
        self.models = []
        for i in self.mon_portefeuille.get_data():
            self.models.append(self.get_model(i))


    def get_model(self, df_titre):
        df = df_titre.head(self.m)
        self.Y = df['Close']
        self.vars.append(df['Close'].var())
        self.X = np.arange(self.m).reshape(self.m,1)
        model = LinearRegression()
        model.fit(self.X,self.Y)
        self.scores.append(model.score(self.X,self.Y))
        self.mon_portefeuille.set_scores(self.scores)
        #print(model.score(self.X, self.Y))
        return model