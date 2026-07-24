
# Conceptes elementals

L'aprenentatge automàtic ens permet abordar tasques massa difícils de resoldre amb programes fixos, escrits i dissenyats per éssers humans. L'aprenentatge profund és un tipus específic d'aprenentatge automàtic que utilitza models compostos per múltiples capes o nivells de processament, encadenats entre si, per aprendre representacions de les dades cada cop més abstractes i complexes, cosa que li permet abordar problemes que els models clàssics no poden resoldre de manera efectiva. Per comprendre'ls bé, cal tenir una sòlida comprensió dels principis bàsics de l'aprenentatge automàtic. Aquest capítol ofereix un breu recorregut pels principis generals més importants que s'apliquen al llarg de tot els materials.


## Models

Un model d'aprenentatge automàtic és aquell que és capaç d'aprendre a partir de dades. En primer lloc és important definir el concepte d'"aprendre" en el nostre context; Mitchell (1997) proporciona una definició concisa i precisa: "Es diu que un programa informàtic (algoritme/model) aprèn de l'experiència _E_ respecte a alguna classe de tasca _T_ i mesura de rendiment _P_, si el seu rendiment en la tasca _T_, mesurada per _P_, millora amb l'experiència _E_." 

Es a dir, podem entendre per model un mètode o algoritme que, a partir de dades d'entrada, és capaç de calcular un valor de sortida (predicció) que realitzi la tasca per a la qual està entrenat. En general, poden ser models estadístics que realitzen estimacions a partir de dades, com ara models lineals, arbres de decisió, màquines de vectors de suport (SVM), entre d'altres. Malgrat que aquests models més simples són perfectament capaços d'abordar problemes senzills de manera apropiada, els problemes del món real no es poden resoldre amb ells. 

En el context d'aquest material parlarem de models d'aprenentatge profund que es diferencien dels enfocaments clàssics principalment pel nombre de paràmetres que han d'ajustar-se i, per aquesta raó, requereixen una gran quantitat de dades d'entrenament. Aquests models consisteixen en múltiples operacions sobre les dades que s'organitzen en diferents capes o nivells encadenats (d'aquí el nom d'aprenentatge profund) per generar les prediccions. Per comprendre millor el seu funcionament, abans d'explicar en detall com funcionen aquests models, en el següent capítol en repassarem alguns dels més senzills, que alhora són la base del funcionament dels models d'aprenentatge profund.


## Conjunt de dades

En el context de l'aprenentatge automàtic, l'experiència, _E_, vé donada pel conjunt de dades que es té disponible. Un conjunt de dades (*dataset*) no és res més que una col·lecció d'exemples. Per treballar-hi, necessitem convertir aquests exemples en una representació numèrica que el model pugui processar. Cada exemple (o mostra) està format per un conjunt de característiques (*features* en angles), a partir de les quals el model genera les seves prediccions.


Una notació habitual per representar un conjunt de dades és la següent:

$$\{(\textbf{x}^{(i)}, \textbf{y}^{(i)}); \ i = 1, \ldots, m\}.$$


On:

- $(\textbf{x}^{(i)}, \textbf{y}^{(i)})$ representa una mostra concreta.
- $\textbf{x}^{(i)} = (x_1^{(i)}, x_2^{(i)}, \ldots, x_n^{(i)})$, amb $x_j^{(i)} \in \mathbb{R}$, és el conjunt de característiques d'aquesta mostra.
- $\textbf{y}^{(i)} = (y_1^{(i)}, y_2^{(i)}, \ldots, y_p^{(i)})$ és el valor esperat (*target*) que el model ha de predir per a aquest conjunt de característiques.

En moltes aplicacions, la predicció consisteix en un únic valor ($\textbf{y}^{(i)} = y^{(i)}$). Aquest valor pot ser:

- Un nombre real, en problemes de **regressió** ($y^{(i)} \in \mathbb{R}$).
- Una categoria, en problemes de **classificació**. En el cas de la classificació binaria es definiria com ($y^{(i)} \in \{0,1\}$).

**La dimensionalitat de les dades**

Si treballéssim amb dades tabulars, XXXXXXX

Si treballéssim amb imatges, cada imatge seria una mostra, representada com una llista ordenada de valors numèrics corresponents al color de cada píxel. Una imatge en color de 400 x 400 píxels ja suposaria 480.000 característiques per mostra. Aquesta xifra il·lustra molt bé la gran dimensionalitat amb què treballen les aplicacions d'aprenentatge automàtic.

Si treballéssim amb text, XXXXXX. **Es liarse molt??**

Com a regla general, com més dades disposem, més fàcil és trobar un model que funcioni bé: podem entrenar models més complexos, amb més paràmetres, sense que caiguin en errors per manca d'informació. L'aparició de repositoris amb grans quantitats de *datasets* públics ha estat clau per a l'avenç de l'aprenentatge automàtic. Aquest fet és encara més cert en el cas del *deep learning*: aquests models necessiten volums de dades molt superiors als dels mètodes d'aprenentatge automàtic clàssics, fins al punt que no funcionen correctament sense grans conjunts de dades. Es podria dir que aquesta dependència de grans *datasets* és, de fet, una de les principals limitacions d'aquest tipus de models.


## Tasques 

Un algoritme d'aprenentatge automàtic es defineix, en part, per la tasca _T_ que ha de resoldre. Aquesta tasca es descriu en termes de com el sistema hauria de processar un exemple, és a dir, un vector de característiques (*features*) que representa un objecte o un esdeveniment del qual volem que el sistema d'aprenentatge automàtic n'extregui informació. Tot seguit es descriuen algunes de les tasques més habituals que es poden resoldre amb aprenentatge automàtic.

### Classificació

En aquest tipus de tasca, es demana al programa que especifiqui a quina de *k* categories pertany una determinada entrada. Per resoldre aquesta tasca, l'algoritme d'aprenentatge sol produir una funció:

$$f: \mathbb{R}^n \rightarrow \{1, ..., k\}.$$

Quan *y = f(x)*, el model assigna una entrada descrita pel vector *x* a una categoria identificada pel codi numèric *y*. Un exemple clàssic és la classificació d'imatges, on l'entrada és una imatge (representada com un conjunt de valors de píxels) i la sortida és la categoria a la qual pertany l'objecte de la imatge (per exemple, "moix", "peix" o "cotxe").

### Regressió

En aquest tipus de tasca, es demana al programa que realitzi una predicció d'un valor numèric a partir d'una determinada entrada. Per resoldre aquesta tasca, l'algoritme d'aprenentatge ha de produir una funció:


$$f: \mathbb{R}^n \rightarrow \mathbb{R}.$$

Aquest tipus de tasca és similar a la classificació, excepte que el format de la sortida és diferent: en lloc d'una categoria discreta, s'obté un valor continu. Un exemple típic és la predicció del preu d'un habitatge a partir de característiques com la superfície, la ubicació o el nombre d'habitacions.

### Transcripció

En aquest tipus de tasca, es demana al sistema d'aprenentatge automàtic que observi una dada relativament no estructurada i que en transcrigui la informació en una forma textual discreta. Per exemple, en el reconeixement òptic de caràcters (*OCR*), es proporciona al model una imatge que conté text i s'espera que el model retorni aquest text en format de seqüència de caràcters. Un altre exemple és el reconeixement de veu, on es proporciona un enregistrament d'àudio i el model ha de produir la seqüència de caràcters o paraules que representa el que s'ha dit.

### Traducció automàtica (*Machine translation*)

En una tasca de traducció automàtica, l'entrada ja consisteix en una seqüència de símbols en un idioma determinat, i el programa ha de convertir-la en una seqüència de símbols en un altre idioma. Aquesta tasca s'aplica habitualment a llenguatges naturals, com ara traduir un text de l'anglès al francès, però també es pot aplicar a altres tipus de seqüències.

### Detecció d'anomalies (*Anomaly detection*)

En aquest tipus de tasca, el model examina un conjunt d'esdeveniments o objectes i n'assenyala aquells que resulten inusuals o atípics respecte a la resta. Un exemple habitual és la detecció de frau amb targetes de crèdit: mitjançant el modelatge dels patrons d'ús habituals d'un client, un sistema d'aprenentatge automàtic pot predir si una transacció determinada és probablement fraudulenta, comparant-la amb el comportament típic de l'usuari.

### Síntesi i mostreig (*Synthesis and sampling*)

En aquest tipus de tasca, es demana a l'algoritme d'aprenentatge automàtic que generi noves mostres similars a les dades d'entrenament. La síntesi i el mostreig mitjançant models d'aprenentatge automàtic poden ser especialment útils en aplicacions de mitjans (*media*), quan generar mostres grans a mà seria costós, tediós o requeriria massa temps. Per exemple, en aplicacions de videojocs es poden generar automàticament textures per a objectes o paisatges grans, en lloc de dissenyar-les manualment o requerir que l'artista n'etiqueti cada píxel.

### Eliminació de soroll (*Denoising*)

En aquest tipus de tasca, es proporciona a l'algoritme d'aprenentatge automàtic un exemple corromput $\tilde{x} \in \mathbb{R}^n$, obtingut a partir d'un exemple net $x \in \mathbb{R}^n$ mitjançant un procés de corrupció desconegut. El sistema ha de predir l'exemple net original *x* a partir de la versió corrompuda $\tilde{x}$, o de manera més general, ha d'estimar la distribució de probabilitat condicional $p(x \mid \tilde{x})$.

### Estimació de densitat (*Density estimation*)

En el problema d'estimació de densitat, es demana a l'algoritme d'aprenentatge automàtic que aprengui una funció:

$$p_{model}: \mathbb{R}^n \rightarrow \mathbb{R},$$

interpretable com una funció de densitat de probabilitat (si l'espai on estan definides les *x* és continu) sobre l'espai del qual s'han extret els exemples d'entrenament. Per fer bé aquesta tasca, l'algoritme ha d'aprendre l'estructura de les dades observades. El model ha de saber en quins punts els exemples s'agrupen densament i en quins punts són poc probables. La majoria de tasques descrites anteriorment requereixen que l'algoritme d'aprenentatge, com a mínim implícitament, capturi l'estructura de la distribució de probabilitat.


Per descomptat, és possible solucionar moltes altres tasques. Aquelles que enumerem aquí tenen com a únic objectiu proporcionar exemples del que pot fer l'aprenentatge automàtic i que anirem treballant durant el curs, no definir una taxonomia rígida i exhaustiva.


## Funció d'avaluació

Per avaluar les capacitats d'un algoritme d'aprenentatge automàtic, es necessari de dissenyar una mesura quantitativa del seu rendiment. Normalment, aquesta mesura de rendiment és específica per a la tasca que està duent a terme el sistema i es coneix amb el nom de funció d'avaluació. Generalment, es defineix com una funció de pèrdua (_loss function_): en lloc de mesurar si el model ho fa bé, mesurem fins a quin punt ho fa malament. 

Com que els models d'aprenentatge automàtic són models matemàtics, calen funcions de pèrdua definides de manera formal que indiquin si l'ajust dels paràmetres és encertat o no. 

En l'aprenentatge automàtic, i en general en els mètodes d'optimització, aquestes funcions es coneixen com a funció objectiu. Com que solen definir-se de manera que com més baix és l'error del model, millor és el resultat, aquestes funcions objectiu reben també el nom de funcions de pèrdua. En els problemes de regressió, on l'objectiu és predir valors numèrics continus, la funció de pèrdua més habitual és l'__error quadràtic__, és a dir, el quadrat de la diferència entre la predicció i el valor desitjat. En els problemes de classificació, en canvi, l'objectiu més comú és minimitzar la taxa d'error: la fracció de mostres en què la predicció del model no coincideix amb l'etiqueta real.

Algunes funcions de pèrdua, com l'error quadràtic, són relativament fàcils d'optimitzar. Les funcions de pèrdua pròpies dels problemes de classificació, en canvi, resulten més difícils d'optimitzar directament, sobretot perquè solen ser complexes i, en alguns casos, difícils de derivar. La derivació és precisament el procediment que permet trobar el mínim d'aquestes funcions, el qual correspon al mínim error possible del model.

## Algorisme d'aprenentatge

L'algoritme d'aprenentatge, permet trobar els paràmetres del model que optimitzen la funció de pèrdua en un problema donat emprant un conjunt de dades. Durant l'aprenentatge, el model troba els millors valors dels paràmetres minimitzant la pèrdua (error) de les prediccions que va realitzant el model durant l'aprenentatge amb les dades etiquetades amb els valors desitjats. 

Minimitzar l'error durant l'aprenentatge no garanteix que el model generalitzi, és a dir, que funcioni correctament amb dades noves. Una pràctica comuna consisteix a dividir les dades disponibles en dos conjunts: les dades d'entrenament (o conjunt d'entrenament), per ajustar els paràmetres del model i les dades de test (o conjunt de test), que es guarda per avaluar el model. A cada pas d'aprenentatge, l'algoritme informa dels errors comesos pel model en ambdós conjunts.

Es podria comparar el rendiment del model durant l'aprenentatge amb els resultats d'un estudiant en les proves d'avaluació continuada d'un curs, on uns bons resultats no sempre garanteixen aprovar l'examen final. Quan un model funciona bé en el conjunt d'entrenament, però no aconsegueix generalitzar a dades no vistes, diem que s'està sobre ajustant (_overfitting_). Aquest fenomen s'explica sovint en termes del compromís entre biaix i variància: un model amb biaix alt és massa simple i no arriba a capturar els patrons de les dades (_underfitting_), mentre que un model amb variància alta és massa complex i s'adapta al soroll específic del conjunt d'entrenament, de manera que els seus resultats varien molt en canviar les dades d'entrenament. L'_overfitting_ és, precisament, la manifestació d'aquesta variància excessiva.

TODO IMATGE: Similar a deep learning 5.2


El tipus d'algoritme d'aprenentatge és la divisió més utilitzada entre els models:

- **Aprenentatge supervisat**: El conjunt de dades està compost per "parells" d'entrenament que consisteixen en una entrada i una sortida desitjada (etiqueta).
- **Aprenentatge no supervisat**: En aquest cas, el conjunt de dades està compost només per les possibles entrades del model. Aquest ha de ser capaç d'agrupar les dades segons l'especificació del problema.

Alguns algoritmes d'aprenentatge automàtic no es limiten a experimentar un conjunt de dades fix. Per exemple, els algoritmes d'aprenentatge per reforç interactuen amb un entorn, de manera que hi ha un bucle de retroalimentació entre el sistema d'aprenentatge i les seves experiències. Aquest tipus d'algoritmes queden fora de l'abast d'aquest curs.




