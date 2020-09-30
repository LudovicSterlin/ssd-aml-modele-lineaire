
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 1 : Utilisation de scikit-learn pour la regression lineaire
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

#generation de donnees test
n = 100
x = np.arange(n)
y = np.random.randn(n)*30 + 50. * np.log(1 + np.arange(n))

# instanciation de sklearn.linear_model.LinearRegression
lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'

# representation du resultat
fig = plt.figure()
plt.plot(x, y, 'r.')
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.legend(('Data', 'Linear Fit'), loc='lower right')
plt.title('Linear regression')
plt.show()


#QUESTION 1.1 : 
#Bien comprendre le fonctionnement de lr, en particulier lr.fit et lr.predict

#QUESTION 1.2 :
#On s'interesse a x=105. En supposant que le model lineaire soit toujours 
#valide pour ce x, quelles valeur corresondante de y vous semble la plus 
#vraisemblable ? 

"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 2 : impact et detection d'outliers
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

#generation de donnees test
n = 10
x = np.arange(n)
y = 10. + 4.*x + np.random.randn(n)*3. 
y[9]=y[9]+20

# instanciation de sklearn.linear_model.LinearRegression
lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'

# representation du resultat

print('b_0='+str(lr.intercept_)+' et b_1='+str(lr.coef_[0]))

fig = plt.figure()
plt.plot(x, y, 'r.')
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.legend(('Data', 'Linear Fit'), loc='lower right')
plt.title('Linear regression')
plt.show()


#QUESTION 2.1 : 
#La ligne 'y[9]=y[9]+20' genere artificiellement une donnee aberrante.
#-> Tester l'impact de la donnee aberrante en estimant b_0, b_1 et s^2 
#   sur 5 jeux de donnees qui la contiennent cette donnee et 5 autres qui
#   ne la contiennent pas (simplement ne pas executer la ligne y[9]=y[9]+20).
#   On remarque que $\beta_0 = 10$, $\beta_1 = 4$ et $sigma=3$ dans les 
#   données simulees.


#QUESTION 2.2 : 
#2.2.a -> Pour chaque variable i, calculez les profils des résidus 
#         $e_{(i)j}=y_j - \hat{y_{(i)j}}$ pour tous les j, ou   
#         \hat{y_{(i)j}} est l'estimation de y_j a partir d'un modele  
#         lineaire appris sans l'observation i.
#2.2.b -> En quoi le profil des e_{(i)j} est different pour i=9 que pour  
#         les autre i
#2.2.c -> Etendre ces calculs pour définir la distance de Cook de chaque 
#         variable i
#
#AIDE : pour enlever un element 'i' de 'x' ou 'y', utiliser 
#       x_del_i=np.delete(x,i) et y_del_i=np.delete(y,i) 




"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 3 : Vers la regression lineaire multiple et optimisation
#
#On considere que l'on connait les notes moyennes sur l'annee de n eleves 
#dans p matieres, ainsi que leur note a un concours en fin d'annee. On 
#se demande si on ne pourrait pas predire la note des etudiants au 
#concours en fonction de leur moyenne annuelle afin d'estimer leurs 
#chances au concours.
#
#On va resoudre le probleme a l'aide de la regression lineaire en 
#dimension p>1 sans utiliser scikit-learn. 
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

#Question 1 :
# - A l'aide de la fonction 'SimulateObservations', simulez un jeu de donnees d'apprentissage [X_l,y_l] avec 30 observations et un jeu de test [X_t,y_t] avec 10 observations. Les observations seront en dimension p=10


def SimulateObservations(n_train,n_test,p):
  """
  n_train: number of training obserations to simulate
  n_test: number of test obserations to simulate
  p: dimension of the observations to simulate
  """
  
  ObsX_train=20.*np.random.rand(n_train,p)
  ObsX_tst=20.*np.random.rand(n_test,p)
  
  RefTheta=np.random.rand(p)**3
  RefTheta=RefTheta/RefTheta.sum()
  print("The thetas with which the values were simulated is: "+str(RefTheta))
  
  ObsY_train=np.dot(ObsX_train,RefTheta.reshape(p,1))+1.5*np.random.randn(n_train,1)
  ObsY_tst=np.dot(ObsX_tst,RefTheta.reshape(p,1))+1.5*np.random.randn(n_test,1)
  
  return [ObsX_train,ObsY_train,ObsX_tst,ObsY_tst,RefTheta]




#Question 2 :
# - On considere un modele lineaire en dimension p>1 mettre en lien les x[i,:] et les y[i], c'est a dire que np.dot(x[i,:],theta_optimal) doit etre le plus proche possible de y[i] sur l'ensemble des observations i. Dans le modele lineaire multiple, theta_optimal est un vecteur de taille [p,1] qui pondere les differentes variables observees (ici les moyennes dans une matiere). Coder alors une fonction qui calcule la moyenne des differences au carre entre ces valeurs en fonction de theta.

def CptMSE(X,y_true,theta_test):
  #TO DO
  
  return MSE



#Question 3 -- option 1 :
# - On va maintenant chercher le theta_test qui minimise cette fonction (il correspondra a theta_optimal), et ainsi résoudre le probleme d'apprentissage de regression lineaire multiple. Utiliser pour cela la fonction minimize de scipy.optimize




#TO DO


#Question 3 -- option 2 :
#De maniere alternative, le probleme peut etre resolu a l'aide d'une methode de descente de gradient codee a la main, dans laquelle les gradients seront calcules par differences finies.




"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 4 : maximum de vraisemblance
# - Tirer 10 fois une piece a pile ou face et modeliser les resultats obtenus comme ceux
#d'une variable aleatoire X qui vaut X_i=0 si on a pile et X_i=1 si on a face.
# - Calculer le maximum de vraisemblance du parametre p d'un loi de Bernoulli qui modeliserait le probleme.
# - Vérifier empiriquement comment évolue ce maximum de vraisemblance si l'on effectue de plus en plus de tirages
# - Que se passe-t-il quand il y a trop de tirages ? Représenter la log-vraisemblance plutot que la vraisemblance dans ce cas.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
