
"""
EXERCICE 7 : Regression logistique
"""


#QUESTION) Il semble qu'un traitement (colonne Treatment) ait un effet sur une variable de "./RealMedicalData2.csv".
#          Utilisez 'sklearn.linear_model.LogisticRegression' pour trouver quelle est cette variable.


dataframe=pandas.read_csv("./RealMedicalData2.csv",sep=';',decimal=b',')

listColNames=list(dataframe.columns)

#...