# GazTorches-Detection & Volume Estimation

Ce projet est une solution d'intelligence artificielle spécialisée pour le secteur industriel (Oil & Gas). Il permet d'automatiser le suivi des émissions en combinant la **vision par ordinateur** pour la détection et le **machine learning** pour l'estimation quantitative du volume de gaz brûlé.



## 📁 Structure du Projet

* **`best.pt`** : Modèle YOLOv8 entraîné et optimisé pour la détection spécifique des flammes de torches.
* **`volume.joblib`** : Modèle de régression (Machine Learning) permettant de prédire le volume à partir des caractéristiques visuelles.
* **`gazTorche.ipynb`** : Notebook complet contenant le prétraitement des données, l'entraînement et l'évaluation des modèles.
* **`pipelien.py`** : Script principal intégrant tout le flux (Pipeline) de l'image brute jusqu'au résultat final.
* **`dataset.csv`** : Base de données structurée liant les dimensions de la flamme au volume réel de gaz.

## 🧠 Techniques Utilisées

### 1. Détection d'Objet (Deep Learning)
Le projet utilise **YOLOv8** (Ultralytics) pour sa rapidité et sa précision en temps réel. Le modèle a été affiné (**Fine-tuning**) pour reconnaître les flammes dans des environnements industriels complexes.

### 2. Ingénierie des Caractéristiques (Features)
Après la détection, le système extrait automatiquement des mesures clés :
- **Largeur ($W$) et Hauteur ($H$)** de la flamme en pixels.
- **Surface (Area)** de la zone détectée.
- **Intensité lumineuse** (pour corréler avec la chaleur de combustion).

### 3. Estimation du Volume (Régression)
Contrairement aux méthodes classiques, ce projet utilise une approche data-driven :
- **Algorithme** : Utilisation de modèles de régression (via `scikit-learn`) pour transformer les caractéristiques visuelles en une valeur numérique de volume (m³/h).
- **Avantage** : Prend en compte les variations non-linéaires de la forme de la flamme.

## 🚀 Installation & Utilisation

1. **Environnement** : Python 3.8+
2. **Dépendances** : 
   ```bash
   pip install ultralytics scikit-learn opencv-python joblib```
3. **Excution** : python pipelien.py --source image_ou_video.mp4
