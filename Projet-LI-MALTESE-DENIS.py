import numpy as np
from matplotlib import pyplot as plt
from numpy import array
import copy

#Fonction donnant la dérivée des données(S,I,T,R)à un moment donné

def f(x) :
    return np.array([(-beta/N)*x[0]*(x[1]+sigma*x[2]),(beta/N)*x[0]*(x[1]+sigma*x[2])-((alpha+gamma)*x[1]),(alpha*x[1])-(nu*x[2]),gamma*x[1]+nu*x[2]])

#Fonction donnant la Jacobienne du système(S,I,T,R) de la fonction f ci-dessus

def df(x):
    return np.array([[(-beta/N)*(x[1]+sigma*x[2]),(-beta/N)*x[0],(-beta/N)*x[0]*sigma,0],[(beta/N)*(x[1]+sigma*x[2]),(beta/N)*x[0]-(alpha+gamma),(beta/N)*x[0]*sigma,0],[0,alpha,-nu,0],[0,gamma,nu,0]])

#Méthode de Newton pour résoudre l'équation posée par le schéma d'Euler implicite

def newton(f,df,x0):
    i = 0
    imax = 365 #Nombre d'itérations maximum
    x = x0 #Notre première suggestion pour la solution du problème
    delta = 1
    temp=1
    while i<imax and delta>10**(-5):
        x1 = copy.deepcopy(x)
        p = -np.dot(np.linalg.inv(df(x)),f(x))#Second membre de l'itération(=g(x)/dg(x))
        x = x + p
        temp=np.linalg.norm(f(x))
        if(temp == 0):
            return x
        delta = np.linalg.norm(f(x)-f(x1))/np.linalg.norm(f(x))#Mise à jour de delta
        i=i+1
    return x

#Utilisation de la Méthode de newton pour simuler l'évolution de l'épidémie

def euler_newton(f,df,y0,step,length):
    h=step #Notre deltaT
    t,y=np.linspace(0,length,length//step+1),[]
    y.append(y0)#On ajoute à la solution totale l'état intitial pour commencer
    for n in range(length//step):
        def g(u): return -u+y[n]+h*f(u) #Equation fournie par le schéma d'Euler implicite
        def dg(u): return -1*np.eye(4) + h*df(u) #dg y[n] est une constante dans g donc elle disparaît
        sol=newton(g,dg,y[n]+f(y[n])) #pour que f soit egale a g
        y.append(sol) #On ajoute le nouveau yn à la solution totale
    return t,y

#Méthode du point fixe

def PointFixe(F,x0):
    x=x0 #Notre première suggestion pour la solution du problème
    i=0
    imax=365 #Nombre d'itérations maximum
    delta=abs(F(x)-x)
    while (delta>10**(-5)).all() and i<imax:
          x=F(x) #Mise à jour du x
          i=i+1
          delta=abs(F(x)-x)
    return x

#Utilisation de la Méthode du point fixe pour simuler l'évolution de l'épidémie

def resfixe(f,y0,step,length):
    h=step #Notre deltaT
    t,y=np.linspace(0,length,length//step+1),[]
    y.append(y0)
    for n in range(length//step):
        def F(u): return y[n]+h*f(u)
        sol=PointFixe(F,y[n])
        y.append(sol) #On ajoute le nouveau yn à la solution totale
    return t,y

#Méthode de résolution du systèmee en utilisant Euler explicite

def euler_exp(f,y0,step,length):
    h=step #Notre deltaT
    t,y=np.linspace(0,length,length//step+1),[]
    y.append(y0)
    for n in range(length//step):
        sol=y[n]+h*f(y[n]) #Obtention de yn+1 à partir de yn
        y.append(sol) #On ajoute le nouveau yn à la solution totale
    return t,y

##Initialisation des paramètres
nbp=float(input("Nombre se personnes total=?"))
inf=float(input("Nombre d'infectés au départ=?"))
length=int(input("Durée de la simulation(jours en entier)=?"))
step=int(input("Durée d'un pas de temps en jours(en entier)=?"))
beta = float(input("beta(taux de propagation)=?"))# β 
gamma = float(input("gamma(taux d'infectiosité)=?"))# γ
nu = float(input("nu(sortie de I)=?")) # η
sigma= float(input("delta(efficacité du traitement)=?"))# δ
alpha = float(input("alpha(taux d'infectés traités)=?"))# α
R0 =beta/(alpha+gamma)+(alpha/(alpha+gamma))*sigma*beta/nu
print("R0(taux de reproduction)= ")
print(R0)
x0=array([nbp-inf,inf,0,0])
N=nbp


#Création du graphique correspondant à la résolution du système
#avec la méthode d'Euler explicite
plt.figure(1)
plt.plot()
plt.title('Euler explicite')
t1,x1=euler_exp(f,x0,step,length)
[o1,o2,o3,o4]=plt.plot(t1,x1)
plt.legend([o1,o2,o3,o4],["S(t)","I(t)","T(t)","R(t)"])
plt.grid()
plt.xlabel('Temps(en jours)')
plt.ylabel('Nombre de personnes')


#Création du graphique correspondant à la résolution du système
#avec la méthode d'Euler implicite avec Newton
plt.figure(2)
plt.plot()
plt.title('Euler Implicite+Newton')
t2,x2=euler_newton(f,df,x0,step,length)
[p1,p2,p3,p4]=plt.plot(t2,x2)
plt.legend([p1,p2,p3,p4],["S(t)","I(t)","T(t)","R(t)"])
plt.grid()
plt.xlabel('Temps(en jours)')
plt.ylabel('Nombre de personnes')

#Création du graphique correspondant à la résolution du système
#avec la méthode du point fixe
plt.figure(3)
plt.plot
plt.title('Euler implicite et Point_fixe')
t3,x3= resfixe(f,x0,step,length)
[q1,q2,q3,q4]=plt.plot(t3,x3)
plt.legend([q1,q2,q3,q4],["S(t)","I(t)","T(t)","R(t)"])
plt.grid()
plt.xlabel('Temps(en jours)')
plt.ylabel('Nombre de personnes')

#Affichage des trois graphiques
plt.show()
