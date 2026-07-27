# El Perceptró

El cervell humà està format per una substància grisa que conté aproximadament 100.000 milions de neurones, cèl·lules nervioses especialitzades en la transmissió d'informació entre el cervell i la resta del cos, interconnectades formant una xarxa de comunicació d'una complexitat extraordinària.

FIGURA 

Com es veu a la figura anterior, cada neurona té múltiples connexions amb altres cèl·lules per mitjà de la sinapsi, que és una reacció química entre el senyal elèctric generat al nucli de la cèl·lula, que quan rep prou connexions per mitjà d'unes terminacions denominades dendrites, es transmet per l'axó cap a les neurones següents.

L'any 1958, al Laboratori Aeronàutic de Cornell, el psicòleg Frank Rosenblatt va crear el **Perceptró** inspirant-se en el funcionament de les neurones (Rosenblatt, 1958). Aquest model es basa en la definició d'una neurona artificial:

El valor de les entrades, $x$, es pondera per uns pesos, $w$, i s'acumula al nucli de la neurona artificial per mitjà de la funció neta:

$$net = \sum_{j=1}^{n} w_j x_j,$$

quan l'entrada neta és major o igual que un valor de llindar, $\theta$, la sortida, $y$, és igual a $1$; si no, la sortida valdria $0$.

Per entendre el seu funcionament, vegem primer un exemple de com funcionaria un Perceptró com una porta lògica AND. Si recordem la definició de la Regressió logística del capítol anterior:

$$\hat{y} = f\left(\sum_{j=0}^{n} w_j x_j\right), \quad \text{on} \quad f(z) = \frac{1}{1+e^{-z}},$$

podríem veure el Perceptró com un model de classificació lineal on la funció d'activació seria la funció llindar en lloc de la Regressió logística:

$$\hat{y} = f\left(\sum_{j=1}^{n} w_j x_j\right), \quad \text{on} \quad f(z) = \begin{cases} 1 & \text{si } z \geq \theta \\ 0 & \text{si } z < \theta \end{cases}$$

Aquesta expressió es pot modificar per convertir el llindar en el valor biax d'aquest model mitjançant la convenció $b = -\theta$, $x_0 = 1$, el model queda de la següent manera:

$$\hat{y} = \begin{cases} 1 & \text{si } \displaystyle\sum_{j=0}^{n} w_j x_j \geq 0 \\ 0 & \text{si no} \end{cases}$$

A més, la regla d'aprenentatge del Perceptró, basada de nou en el descens del gradient, és:

$$w_j = w_j + \eta (y - \hat{y}) \cdot x_j, $$

que, de nou, conceptualment és la mateixa regla que la dels models lineals vistos anteriorment. A partir del model teòric del Perceptró, es va construir el Perceptron Mark 1, un ordinador dissenyat per al reconeixement d'imatges que tenia una matriu de 400 fotocèl·lules connectades aleatòriament a una implementació electrònica de les neurones artificials; els pesos es codificaven en potenciòmetres i les actualitzacions dels pesos durant l'aprenentatge es realitzaven mitjançant motors elèctrics.

> *The New York Times* va informar que el perceptró era "l'embrió d'un ordinador electrònic que [la Marina] espera que sigui capaç de caminar, parlar, veure, escriure, reproduir-se i ser conscient de la seva existència."
>
> Mikel Olazarán (1996). "A Sociological Study of the Official History of the Perceptrons Controversy". *Social Studies of Science*. 26(3): 611–659.

Tanmateix, malgrat l'expectació generada per aquest model, la publicació del llibre *Perceptrons* (Minski and Papert, 1969) va provocar un abandonament en el finançament de la recerca en xarxes neuronals artificials. En aquest llibre es demostrava que el Perceptró no era capaç de resoldre un problema senzill com el de la porta lògica XOR.

FIGURAAA

Com podem veure a la figura, s'arriba a una contradicció en la derivació matemàtica del problema. Si ens fixem en la seva representació gràfica, podem comprovar com no és possible separar els exemples de les dues classes mitjançant una recta.

En general, els models lineals de classificació, com el Perceptró, només poden resoldre aplicacions de classificació on els exemples de cada classe puguin separar-se per un hiperplà (una recta $n$-dimensional). Tanmateix, si ens fixem en la representació gràfica de la funció XOR, si tinguessim la capacitat de definir dues rectes de separació, és a dir, definir dos Perceptrons podríem resoldre aquest problema de manera més senzilla.

De nou, fixant-nos el comportament del cervell humà, les neurones s'organitzen en capes i, per tant, aquesta limitació desapareix amb un **Perceptró multicapa**. En aquest cas, el problema era que fins al 1986 no es va trobar una regla d'aprenentatge per al Perceptró multicapa: l'algoritme Backpropagation (Rumelhart, Hinton and Williams, 1986), que veurem en el proper tema.

## El Perceptró multicapa

El Perceptró que hem vist fins ara modela el comportament d'una única neurona, capaç de realitzar una separació lineal de l'espai de característiques. Tanmateix, com hem comprovat amb l'exemple de la porta lògica **XOR**, aquesta limitació el fa incapaç de resoldre problemes que no siguin linealment separables. De la mateixa manera que el cervell humà no funciona amb una única neurona aïllada sinó amb milers de milions d'elles organitzades en estructures complexes, la solució natural a aquesta limitació és connectar múltiples Perceptrons entre si, formant el que es coneix com a Perceptró multicapa o xarxa neuronal.

A continuació tenim l'esquema general del perceptró multicapa:

<figure style="text-align: center;">
    <img src="../assets/xarxa_neuronal_cat.png" width="400"
         alt="Xarxa neuronal multi capa.">
    <figcaption>Xarxa neuronal multi capa.</figcaption>
</figure>

En aquesta figura, hem usat cercles blaus per indicar les entrades a la xarxa. Els cercles etiquetats com a "+1" són les unitats associades al valor del biaix, que ja havíem vist en els models lineals. La capa més a l'esquerra de la xarxa es denomina **capa d'entrada**, i la capa més a la dreta, la **capa de sortida** que, en aquest exemple, només té una neurona. La capa intermèdia de neurones es denomina **capa oculta**, perquè els seus valors no són observables durant l'entrenament. Per definir l'arquitectura del nostre model podem dir que la nostra xarxa neuronal té 3 unitats d'entrada (sense comptar la unitat de biaix), 3 unitats ocultes i 1 unitat de sortida.

### Notació formal

Si $n_l$ és el nombre de capes a la nostra xarxa, etiquetarem la capa $L$ com $L_l$, de manera que la capa $L_1$ seria la capa d'entrada i la capa $L_{n_l}$ la capa de sortida ($L_3$ en el nostre exemple).

Així, escrivim $W_{ij}^{(l)}$ per indicar el pes (o paràmetre) associat amb la connexió entre la unitat $j$ de la capa $l$ i la unitat $i$ a la capa $l+1$. A més, $b_i^{(l)}$ és el bias associat a la unitat $i$ de la capa $l+1$. En aquest cas, tal com es pot observar a la figura, les unitats associades als bias no tenen connexions amb la capa anterior. Finalment, escriurem $a_i^{(l)}$ per indicar el valor d'activació (és a dir, el valor de sortida) de la unitat $i$ a la capa $l$ (així, usaríem $a_i^{(1)} = x_i$ per indicar la $i$-èsima entrada). Donada una configuració de paràmetres $\{\mathbf{W}^{(l)}, \mathbf{b}^{(l)}\}, l = 1, \ldots, n_l - 1$, la nostra xarxa neuronal defineix una hipòtesi $h_{\mathbf{W},\mathbf{b}}$ que és el valor de sortida de la xarxa (un nombre real).

Utilitzarem l'exemple de l'arquitectura per definir com es calcula el valor de sortida de la xarxa neuronal, el que es coneix com a **propagació cap endavant** (*forward propagation*). Si definim $z_i^{(l)}$ com la suma total ponderada de les entrades a la unitat $i$ a la capa $l$ (inclòs el bias), tindríem el valor d'entrada neta del Perceptró, de manera que per calcular el valor de sortida de la unitat, només caldria aplicar la funció d'activació, $f$, de manera que seria $a_i^{(l)} = f(z_i^{(l)})$.

En general, recordant que també usem $a_i^{(1)} = x_i$ per denotar els valors de la capa d'entrada, donades les activacions de la capa $l$, $a_i^{(l)}$, podem calcular les activacions de la capa $l+1$, $a_i^{(l+1)}$, com:

$$z_i^{(l+1)} = \sum_{j=1}^{s_l} W_{ij} \, a_j^{(l)} + b_i^{(l)}$$

$$a_i^{(l)} = f\!\left(z_i^{(l)}\right)$$

on $s_l$ és el nombre d'unitats de la capa $l$, i $f$ és la funció d'activació. Aplicant de manera recursiva aquestes fórmules podríem calcular el valor de les unitats de sortida de qualsevol arquitectura de xarxa neuronal. Tots aquests càlculs es poden realitzar de manera paral·lela com a producte de matrius i vectors, que podríem generalitzar usant el concepte de **tensor**, origen de l'expansió d'aquests models gràcies a l'aparició de plataformes de maquinari (com les GPU de NVIDIA) i programari (com TensorFlow) que faciliten l'aplicació i la programació d'aquests models.

Podria semblar que ja sabem definir i usar una xarxa neuronal, però si ens hi fixem bé, encara ens queda una cosa per comentar: la funció d'activació. Aquest concepte és clau en les aplicacions d'aprenentatge profund. De nou, de manera similar als models lineals de classificació, la funció d'activació més utilitzada a la capa de sortida de la xarxa neuronal és la funció sigmoide quan el valor de sortida desitjat està entre 0 i 1.

**1. Sigmoide:**

$$f(z) = \frac{1}{1+e^{-z}}, \qquad f'(z) = f(z) \cdot (1 - f(z))$$

Com es pot veure, la seva derivada és senzilla i queda expressada en forma de la pròpia funció original (aquest fet estalvia càlculs, ja que la derivada de la funció d'activació serà necessària en la regla d'aprenentatge basada en el descens del gradient). Una altra opció similar, però quan el valor de sortida està entre $-1$ i $1$, és la funció tangent hiperbòlica, o *tanh*.

**2. Tangent hiperbòlica:**

$$f(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}, \qquad f'(z) = 1 - f^2(z)$$

Les dues funcions anteriors són útils quan tenim una sola sortida en un problema de classificació binària; tanmateix, en un problema de classificació en múltiples categories, sol usar-se una unitat de sortida que s'activa per a cada categoria, i per tant, tindríem tantes unitats de sortida com classes o categories. En aquest cas, sol usar-se la funció d'activació *softmax*, el valor de la qual depèn dels valors de totes les unitats de sortida i és una generalització de la funció logística per al cas multiclasse. Dit de manera més senzilla, associa probabilitats a cada classe, i per tant, podem saber la probabilitat que l'entrada pertanyi a cadascuna de les categories de sortida; si haguéssim d'escollir, la sortida seria la de més probabilitat.

**3. Softmax:**

$$f(z)_i = \frac{e^{z_i}}{\displaystyle\sum_{k=1}^{K} e^{z_k}}, \quad \text{per a } i = 1, \ldots, K, \quad \mathbf{z} = (z_1, \ldots, z_K)$$

Tanmateix, recerques recents van trobar una funció d'activació diferent, la **funció lineal rectificada** (*ReLU*), que funciona millor a la pràctica a les capes ocultes de les xarxes neuronals profundes (de moltes capes). Aquesta funció d'activació és diferent de la sigmoide i la *tanh* perquè no està limitada ni és contínuament diferenciable, que és una de les condicions "teòriques" que han de complir aquestes funcions per poder ser usades en el descens del gradient.

**4. ReLU:**

$$f(z) = \max(0, z) = \begin{cases} z & \text{si } z > 0 \\ 0 & \text{si no} \end{cases}, \qquad f'(z) = \begin{cases} 1 & \text{si } z > 0 \\ 0 & \text{si } z < 0 \end{cases}$$

Com podem veure, en el cas de $z = 0$, la derivada pot ser $0$ o $1$ indistintament.

## Resum

En aquest apartat hem generalitzat el model de xarxa neuronal (Perceptró multicapa), model fonamental de l'aprenentatge profund. De manera independent del nombre de capes i d'unitats per capa, s'ha explicat com es realitza la notació d'aquests models per poder comprendre les funcions necessàries perquè realitzin les seves prediccions. El concepte bàsic més important que hem introduït és el de capes i unitats ocultes (les que no són entrada ni sortida). També s'ha explicat la relació de les xarxes neuronals amb el concepte de "tensor", que permet accelerar els seus càlculs i, finalment, els diferents tipus de funcions d'activació més utilitzades.

---

# L'algoritme *backpropagation*

En l'explicació del Perceptró vam veure que, tot i que era conegut que els problemes linealment no separables, com el de la porta lògica XOR, es podien solucionar amb una xarxa neuronal multicapa, no va ser fins al 1986 que es va definir un algoritme d'aprenentatge per a aquest tipus de models: l'algoritme *backpropagation*. I quin era el problema que impedia aplicar el descens del gradient als models multicapa? Doncs, senzillament, que només és possible calcular la funció de pèrdua a l'última capa, ja que allà tenim el valor de sortida de la xarxa neuronal i el valor de sortida desitjat, i per tant, podem comparar-los. Però, quin seria el valor desitjat d'una unitat de les capes ocultes?

La idea bàsica de l'algoritme *backpropagation* és que és possible calcular el valor de la funció de pèrdua de les unitats de l'última capa i propagar-lo cap enrere, fins a arribar a les unitats de la primera capa oculta.

Així, l'algoritme seria el següent:

Fer un pas de propagació cap endavant, calculant les activacions per a totes les unitats de la xarxa neuronal fins a arribar a la capa de sortida, $L_{n_l}$.

Per a cada unitat de sortida $i$ de la capa $n_l$ (la capa de sortida), calculem el valor de la funció de pèrdua per la derivada de la funció d'activació, que correspondrà a l'error comès per la unitat $i$ de sortida, $\delta_i^{(n_l)}$. Així, per exemple, per a la funció de pèrdua de la diferència de l'error al quadrat, o la d'entropia creuada, seria:

$$\delta_i^{(n_l)} = -\left(y_i - a_i^{(n_l)}\right) \cdot f'\!\left(z_i^{(n_l)}\right)$$

Per a la resta de capes, des de la capa $n_l - 1$ fins a la capa $1$, propagaríem els termes d'error cap enrere:

$$\delta_i^{(l)} = \left(\sum_{k=1}^{s_{l+1}} W_{ki}^{(l)} \, \delta_k^{(l+1)}\right) \cdot f'\!\left(z_i^{(l)}\right)$$

Finalment, una vegada calculats tots els errors de totes les unitats, les regles d'ajust de tots els pesos i bias de la xarxa neuronal serien:

$$W_{ij}^{(l)} = W_{ij}^{(l)} - \alpha \, a_j^{(l)} \, \delta_i^{(l+1)}$$

$$b_i^{(l)} = b_i^{(l)} - \alpha \, \delta_i^{(l+1)}$$

A la figura següent es pot veure de manera gràfica com el càlcul dels termes d'error de cada unitat es calcula de manera inversa al càlcul de la predicció, començant per l'error a l'última capa (que podem calcular fàcilment a partir de la funció de pèrdua) fins als termes d'error de la primera capa oculta.

Para ver la derivación completa del cálculo de la regla de aprendizaje
podéis consultar el siguiente
\href{https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd}{enlace}.
