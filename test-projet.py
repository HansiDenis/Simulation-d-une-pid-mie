# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:16:17 2020

@author: lolo_
"""
import numpy as np
import copy
import matplotlib.pyplot as plt

# =============================================================================
# Fonctions globales statiques 
# =============================================================================

#Il s'agit de la fonction présent dans la relation de la méthode euler implicite:
#Xn+1 = Xn + deltaT * f(Xn+1) 
def f_euler(x):
    y=np.zeros((4,1))
    y[0]=(-beta/N)*x[0]*x[1]+(-beta/N)*x[0]*delta*x[2]
    y[1]=(beta/N)*x[0]*x[1]+(beta/N)*x[0]*delta*x[2]-alpha*x[1]-gamma*x[2]
    y[2]=alpha*x[1]-eta*x[2]
    y[3]=gamma*x[1]+eta*x[2]
    return y

#Il s'agit de la fonction présent dans la relation de la méthode de Newton:
#f(Xn+1)=0
#Cette fonction dépend des valeurs Xn=(Sn,In,Tn,Rn) calculé à l'étape précédente
#d'où la présence d'un deuxième paramètre, celui-ci changera à chaque étape
def f_newton(x,Xn):
    y=np.zeros((4,1))
    y[0]=x[0]-Xn[0]+(beta/N)*x[0]*x[1]*deltaT+(beta/N)*x[0]*delta*x[2]*deltaT
    y[1]=x[1]-Xn[1]+(-beta/N)*x[0]*x[1]*deltaT+(-beta/N)*x[0]*delta*x[2]*deltaT+alpha*x[1]*deltaT+gamma*x[2]*deltaT
    y[2]=x[2]-Xn[2]-alpha*x[1]*deltaT+eta*x[2]*deltaT
    y[3]=x[3]-Xn[3]-gamma*x[1]*deltaT-eta*x[2]*deltaT
    return y

#Il s'agit de la jacobienne fonction présent dans la relation de récurrence de la méthode de Newton:
#Xn+1 = Xn + inv(Jf(Xn))*f(Xn)
def df_newton(x):
    y=np.zeros((4,4))
    y[0,0]=1+(beta/N)*x[1]*deltaT+(beta/N)*delta*x[2]*deltaT
    y[0,1]=(beta/N)*x[0]*deltaT
    y[0,2]=(beta/N)*x[0]*deltaT*delta
    y[1,0]=(-beta/N)*x[1]*deltaT+(-beta/N)*delta*x[2]*deltaT
    y[1,1]=1-(beta/N)*x[0]*deltaT+alpha*deltaT+gamma*deltaT
    y[1,2]=(-beta/N)*x[0]*deltaT*delta
    y[2,1]=-alpha*deltaT
    y[2,2]=1+eta*deltaT
    y[3,1]=-gamma*deltaT
    y[3,2]=-eta*deltaT
    y[3,3]=1
    return y

# =============================================================================
# Définition des variables globales (paramètres, conditions initiales, méthodes ...)
# =============================================================================

N = 1000        #taille de la population
beta =  1.3      #nbre de rencontres d'un invidu en moyenne
delta =  0.2   #facteur de réduction de l'infectivité d'une personne aprés traitement 
alpha =  0.1    #fraction d'invidus I sélectionnés pour traitement par unité de temps
gamma = 0.1    #taux de guérison d'individus I par unité de temps
eta =  0.1      #taux d'immunité/décés aprés traitement par unité de temps

S0 = 999
I0 = 1
T0 = 0
R0 = 0 
y0 = np.array([[S0],
               [I0],
               [T0],
               [R0]])

duree = 100         #période de simulation
n = 100                 #le nombre de subdivisions de l'intervalle [0,durée], correspond au nombre d'itérations de notre boucle d'euler implicite
deltaT = (duree-0)/n    #ceci sera notre pas temporelle 

eps=1e-12   #epsilon de la méthode de newton
Nmax=100    #nombre de répétitions pour la méthode de newton


# =============================================================================
# Programme principale 
# =============================================================================


def main():   
    res = np.array(y0)  #à la fin de la résolution ce sera donc une matrice 4*(n+1) car n itérations + la CI
    #t correspond à Yn+1, il stockera le résultat calculé par euler implicite Xn+1=(Sn+1,In+1,Tn+1,Rn+1) à chaque étape n
    t=y0                #ainsi t commence à la CI (condition initiale)
    for i in range(1,n+1): #boucle pour la relation euler implicite : période de simulation temporelle
    
        #d'abord on calcule Xn+1=(Sn+1,In+1,Tn+1,Rn+1) avec newton 
        x=t #le x0 de newton est à chaque fois le dernier Xn calculé par euler implicite (ici en l'occurence t)
        k=0
        erreur=1.0 
        while(erreur>eps and k<Nmax):
              xold=copy.deepcopy(x)
              
              #le x calculé ci-dessous est une première valeur de Xn=(Sn,In,Tn,Rn) par la méthode de newton 
              x=x-np.dot(np.linalg.inv(df_newton(x)),f_newton(x,t))  # là au lieu de inv() faut utiliser une méthode utilisée dans le cours
              
              erreur=np.linalg.norm(x-xold)/np.linalg.norm(x)
              k=k+1
        
        #ici, la méthode de newton nous a donné une première valeur de Xn+1=(Sn+1,In+1,Tn+1,Rn+1) stockée dans x
        #on pourrait s'arrêter et passer à l'étape suivante mais on injecte cette valeur dans la relation de 
        #Euler implicite pour avoir une meilleure précision
              
        t = t + deltaT * f_euler(x) #on calcule Xi+1 avec euler implicite, ici x est la valeur retournée par la méthode de newton
        
        res = np.concatenate((res,t),1)   #on ajoute t au vecteur résultat

        
    return res # chaque ligne correspond aux valeurs S I T R , et les colonnes sont les Xn+1=(Sn+1,In+1,Tn+1,Rn+1) calculés

v = main()

#Mettre tous les valeurs supérieures à N égale à N
#Mettre tous les valeurs inférieures à N ègal à 0

R0 = beta/(alpha + gamma) + alpha/(alpha + gamma)*delta*beta/eta
print(R0)
if (R0 > 1):
    print("On a une épandémie")
else:
    print("ta grams")

# =============================================================================
# Graphe représentant chaque variable S,I,R,T
# =============================================================================

temps = np.linspace(0,duree,n+1) #axe des abcisees

S=v[0,:]
I=v[1,:]
T=v[2,:]
R=v[3,:]

leplot = plt.figure(1)
plt.plot(temps,S,'r',lw=3 , label='S')
plt.plot(temps,I,'g',lw=3 , label='I')
plt.plot(temps,R,'y',lw=3 , label='R')
plt.plot(temps,T,'b',lw=3 , label='T')
leplot.legend()
plt.xlabel('temps')
plt.ylabel('habitutants')
plt.show()