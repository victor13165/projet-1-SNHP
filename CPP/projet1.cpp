#include <iostream>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#ifndef _OPENMP_
#include <omp.h>
#endif

#define PI 3.1415926

using namespace std;

//Pour print un vecteur dans un fichier déjà ouvert sous forme de ofstream
void print_tab_in_file(double *x, int N, ofstream &file) {

	for (int i=0;i<N-1;i++) {
		file << x[i]<<",";
	}
	file << x[N-1] << endl;

}

/*
Fonction linspace pour initialiser un vecteur entre start et stop. Dans le cadre du projet, start = -1 et stop = 1
Paramètres :
	- x : le vecteur qui sera rempli
	- num : nombre de points, qui doit aussi être la taille du vecteur x!!
	- start : borne inférieure du domaine
	- stop : borne supérieure du domaine
*/
void linspace(double *x, int num, double start, double stop) {
	int i;
	double dx = (stop-start)/(double) (num-1);

	#pragma omp parallel for private(i) shared(x,dx,num) //Parallélisation
	for (i=0; i<num; i++) {
		x[i] = start + (double) i*dx;
	}
}

/*

Fonction pour initialiser un vecteur selon la condition initiale f(x,t=0) = 2+cos(pi*x)
Paramètres :
	- x : le vecteur qui sera rempli
	- N : nombre de points, qui doit aussi être la taille des vecteurs x et f0
	- start : borne inférieure du domaine par défaut à -1
	- stop : borne supérieure du domaine par défaut à 1
*/
void init(double *x, double *f0, int N, double start=-1.0, double stop=1.0) {
	int i;
	linspace(x,N,start,stop); //Initialiser le vecteur x [start,stop]

	#pragma omp parallel for private(i) shared(f0,x,N)
	for (i=0; i<N; i++) {
		f0[i]=2.+cos(PI*x[i]); //Remplir le vecteur de condition initiale
	}
}

/*
Calcule le second membre de l'équation de d'advection-diffusion selon les schémas aux différences finies donnés dans le sujet de projet.

A partir des schémas aux différences finies, le second membre peut s'écrire:

	F(f(x,t)) = c1*(f(i-1) - f(i)) + c2*(f(i+1) - f(i))

Avec c1 = V/dx + c2 et c2 = D/dx²

Note: cela réduit le nombre d'opérations de 6 à 5, c'est déjà ça de gagné

On rappelle que les conditions limites sont périodiques c'est-à-dire f(-1,t) = f(N,t) et f(N+1,t)=f(0,t).

Paramètres :
	- f : vecteur sur lequel appliquer le second membre (à un instant t fixé, f[i] contient les valeurs en espace
	- N : taille du vecteur f
	- t : indice de temps sur lequel calculer le second membre
	- V : vitesse d'advection
	- D : coefficient de diffusion
*/
void smb(double *f, double *sec_membre, int N, double V, double D, double dx) {
	int i;
	double c2=D/(dx*dx), c1=c2+V/dx; //Coefficients constants donc calculés en dehors de la boucle

	//Conditions aux limites
	sec_membre[0] = c1*(f[N-1]-f[0])+c2*(f[1]-f[0]);
	sec_membre[N-1] = c1*(f[N-2]-f[N-1]);

	//On peut paralléliser parce que l'on n'écrit que dans smb[i], les f[i-1] et f[i+1] ne posent pas problème puisque le vecteur f
	//est seulement utilisé en lecture, donc pas d'accès concurrent
	#pragma omp parallel for shared(N,f,sec_membre) firstprivate(c1,c2) private(i) schedule(static)
	for (i=1;i<N-1;i++) { //i va de 1 à N-2, on ne touche pas aux extremités --> conditions limites
		sec_membre[i] = c1*(f[i-1]-f[i]) + c2*(f[i+1]-f[i]);
	}

}

/*
Intègre l'équation d'advection-diffusion en temps selon le schéma Euler explicite
Paramètres:
	- f : Matrice N de la solution au temps t. On réécrit par dessus cette solution à chaque itération
	- smb : vecteur de second membre (normalement calculé avec smb())
	- N : taille du vecteur smb (et nombre de lignes de f)
	- dt : pas d'intégration temporelle
*/
void integre(double *f, double *smb, int N, double dt) {
	int i;

	#pragma omp parallel for private(i) shared(f,dt,N,smb) schedule(static)
	for (i=0;i<N;i++) {
		f[i] = f[i] + dt*smb[i];
	}
}


int main() {

	ofstream file("resultats.csv"); //Fichier dans lequel on va stocker les resulats intermédiaires (augmente considérablement le temps de calcul)
	int N=21,t,i,T;// N: nombre de points de discrétisation. T: nombre de points
	double *x, *f, *sec_membre; //Vecteur d'espace x, solution f et second membre tous le strois des vecteurs de taille N
	double dt=0.0001, tmax=0.1, V = 1.5, D = 0.1, dx = 2./(double)(N-1),alpha; //Pas de temps, temps maximal, Vitesse d'advection, coefficient de diffusion, pas de discrétisation en espace
	T = tmax/dt; //Nombre de points de discrétisation en temps

	alpha = -V/dx+D/(dx*dx); //Coefficient pour étudier la stabilité

	if (abs(dt*alpha+1.)>1. || abs(dt*alpha) < 0.0000000000000001 ) { //Le schéma est instable si |dt*alpha+1| >= 1
			cout << "Condition de stabilité non respectée! Solutions possibles: \n--> Diminuez le pas de temps\n--> Faites bien attention à ce que V/dx =/= D/dx²" << endl;
	} else {

		x = new double[N]; //Vecteur x d'espace qui ira de -1 à 1
		sec_membre = new double[N]; //vecteur qui va stocker le second membre
		f = new double[N]; //Vecteur qui va stocker la solution au temps final auquel on aura intégré

		init(x,f,N); //Initialisation du vecteur x ainsi que du vecteur condition initiale
		print_tab_in_file(x,N,file); //Vecteur x
		for (t=0;t<T;t++) {
			print_tab_in_file(f,N,file); //Print dans le fichier. Cette opération augmente beaucoup le temps de calcul
			smb(f,sec_membre,N,V,D,dx); //calcul second membre
			integre(f,sec_membre,N,dt); //intégration de tn à tn+1
		}

		file.close(); //Fermer le fichier

		//Libérer l'espace mémoire
		delete [] x;
		delete [] f;
		delete [] sec_membre;
	}
	return 0;
}
