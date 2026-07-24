# Descens de gradient (*Gradient Descent*)

El descens de gradient és un algoritme d'optimització iteratiu utilitzat per trobar el mínim (local o global) d'una funció, generalment una funció de pèrdua o de cost en el context de l'aprenentatge automàtic.

## Definició formal

Sigui $J(\theta)$ una funció de cost (o pèrdua) diferenciable, on $\theta \in \mathbb{R}^n$ representa el vector de paràmetres del model que volem optimitzar. L'objectiu és trobar:

$$\theta^* = \arg\min_{\theta} J(\theta)$$

El descens de gradient assoleix aquest objectiu actualitzant iterativament els paràmetres en la direcció oposada al gradient de la funció de cost, ja que el gradient $\nabla_\theta J(\theta)$ indica la direcció de màxim creixement de la funció. Movent-nos en la direcció oposada, ens acostem a un mínim.

La regla d'actualització dels paràmetres a cada iteració $t$ és:

$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta J(\theta_t)$$

on:

- $\theta_t \in \mathbb{R}^n$ són els paràmetres del model en la iteració $t$
- $\eta > 0$ és la **taxa d'aprenentatge** (*learning rate*), un hiperparàmetre que controla la mida del pas a cada actualització
- $\nabla_\theta J(\theta_t)$ és el gradient de la funció de cost respecte als paràmetres, avaluat en $\theta_t$

## Component a component

De manera més explícita, per a cada component $\theta_i$ del vector de paràmetres:

$$\theta_i \leftarrow \theta_i - \eta \, \frac{\partial J(\theta)}{\partial \theta_i}$$

## Condició d'aturada

L'algoritme s'atura quan es compleix algun criteri de convergència, per exemple:

$$\|\nabla_\theta J(\theta_t)\| < \epsilon$$

on $\epsilon$ és un llindar (*threshold*) petit predefinit, o bé quan s'assoleix un nombre màxim d'iteracions.

## Variants habituals

- **Batch gradient descent:** el gradient es calcula utilitzant tot el *dataset* d'entrenament a cada iteració.
- **Stochastic gradient descent (SGD):** el gradient es calcula utilitzant només un exemple (o una mostra petita) a cada iteració.
- **Mini-batch gradient descent:** un compromís entre les dues opcions anteriors, utilitzant un subconjunt (*batch*) de mides moderades a cada iteració.
