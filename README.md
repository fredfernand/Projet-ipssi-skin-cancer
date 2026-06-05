# 🔬 Skin Cancer Classification — HAM10000

## Auteurs
**FREDERIC FERNANDES DA COSTA, Gills Daryl KETCHA NZOUNDJI JIEPMOU, Narcisse Cabrel TSAFACK FOUEGAP**

## Description
Projet complet de Machine Learning et Deep Learning pour la classification de lésions cutanées en
**7 classes diagnostiques** à partir du dataset Kaggle
[Skin Cancer MNIST HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

> **Classes :** `akiec` · `bcc` · `bkl` · `df` · `mel` · `nv` · `vasc`  
> **Priorité médicale :** Recall du mélanome (`mel`) — faux négatif = cancer non détecté = danger vital

---

## Résultats

| Modèle | Type | F1-weighted | Accuracy | Recall mélanome | Temps |
|--------|------|:-----------:|:--------:|:---------------:|-------|
| LR L1 (Lasso) | ML | 0.6640 | 0.6533 | 0.24 | ~60s CPU |
| CNN v2 (optimisé) | DL | 0.7098 | 0.6693 | 0.62 | 486s GPU |
| **EfficientNetB0** | **Transfer Learning** | **0.7773** | **0.7497** | **0.64** | 1842s GPU |

> Le Transfer Learning améliore le F1 de **+17%** et le recall du mélanome de **+167%** par rapport au ML classique.

---

## Structure du projet
```
skin-cancer-ham10000/
├── skin_cancer_ham10000.ipynb   ← Notebook principal (9 jalons, 112 cellules)
├── src/                         ← Modules créés par %%writefile
│   ├── .gitkeep
│   ├── data_wrangling.py        (créé à l'exécution)
│   ├── eda_utils.py             (créé à l'exécution)
│   ├── ml_models.py             (créé à l'exécution)
│   ├── evaluate.py              (créé à l'exécution)
│   ├── dl_models.py             (créé à l'exécution)
│   ├── dl_advanced.py           (créé à l'exécution)
│   └── dashboard.py             (créé à l'exécution)
├── best_model.h5                ← Modèle EfficientNetB0 sauvegardé
├── README.md
├── requirements.txt
└── .gitignore
```

> ⚠️ **Le dataset n'est PAS versionné** dans ce dépôt (voir `.gitignore`).
> Il est téléchargé automatiquement via `kagglehub` dans le notebook.

> Les fichiers `src/*.py` sont créés à l'exécution du notebook via des cellules
> `%%writefile`. Le dossier `src/` est gardé vide dans le repo (`.gitkeep`).

---

## Jalons

> 🏷️ **Tags Git disponibles** : Tous les jalons sont tagués avec les labels requis (`data`, `eda`, `ml`, `eval-ml`, `dl`, `opti-dl`, `eval-dl`, `deploy`).  
> Consultez-les via `git tag -l` ou sur GitHub dans l'onglet **Releases/Tags**.

### Note 1 — Data Science & Baseline ML

| Jalon | Contenu | Label Git |
|-------|---------|-----------|
| **1 — Data** | Téléchargement, nettoyage (57 NaN imputés, 2545 doublons supprimés), vérification intégrité images | `data` |
| **2 — EDA** | Distribution 7 classes (déséquilibre 74×), métadonnées cliniques (âge, sexe, localisation), analyse pixels, data augmentation | `eda` |
| **3 — ML Classique** | PCA (4096→73 dims), LR L1/L2/ElasticNet, SVM, RF, KNN, régularisation CV 5-fold, features images vs images+métadonnées | `ml` |
| **4 — Éval ML** | Métriques F1-weighted, confusion 7×7, ROC OvR multiclasse, learning curves biais/variance, conclusion mélanome | `eval-ml` |

### Note 2 — Deep Learning Fondamental

| Jalon | Contenu | Label Git |
|-------|---------|-----------|
| **5 — Architecture** | CNN 3 blocs (Conv2D + BatchNorm + ReLU + MaxPool), justification CNN vs MLP vs RNN, stratégie anti vanishing gradient | `dl` |
| **6 — Optimisation** | Recherche d'hyperparamètres (5 configs : LR, dropout, filtres), choix optimiseur Adam, CNN v2 optimisé (lr=5e-4, drop=0.3, f=64) | `opti-dl` |
| **7 — Comparaison** | Tableau ML vs DL, analyse critique (spatialité, features apprises, coût GPU), recall mélanome ×2.6 | `eval-dl` |

### Note 3 — Deep Learning Avancé & Ingénierie

| Jalon | Contenu | Label Git |
|-------|---------|-----------|
| **8 — DL Avancé** | Transfer Learning EfficientNetB0 pré-entraîné ImageNet, fine-tuning 20 couches, F1=0.7773 | `eval-dl` |
| **9 — Dashboard** | Streamlit interactif, frameworks OIA (Observation→Insight→Action) et Pyramide de Minto | `deploy` |

---

## Dashboard Streamlit

Le dashboard utilise deux frameworks de communication professionnels :

- **Pyramide de Minto** : conclusion d'abord → arguments → données détaillées
- **Framework OIA** : chaque résultat répond au "So What?" via Observation → Insight → Action

### Pages du dashboard
| Page | Contenu |
|------|---------|
| 🏠 Accueil | Message clé (Minto), KPIs, tableau comparatif, analyse OIA |
| 🔍 Prédiction | Upload image → classification + probabilités + recommandation OIA |
| 📊 Comparaison | ML vs CNN vs Transfer Learning en format OIA |
| ℹ️ À propos | Dataset, auteurs, architecture, transparence IA |

### Lancer le dashboard
```bash
streamlit run src/dashboard.py
```

> ⚠️ **Modèle inclus dans Git** (`best_model.h5`, 30 MB)  
> Le modèle EfficientNetB0 fine-tuné est versionné dans le dépôt.
> Pour réentraîner : exécuter le notebook jusqu'à la cellule "model.save('best_model.h5')".

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Exécution

### Partie ML (Jalons 1-4)
Ouvrir `skin_cancer_ham10000.ipynb` dans **Google Colab** → kernel **High RAM, CPU**.

### Partie DL (Jalons 5-9)
Même notebook → changer le runtime en **T4 GPU** (Runtime → Change runtime type).

Les modules `src/*.py` sont créés automatiquement par les cellules `%%writefile`.

---

### Outils utilisés
| Catégorie | Outils | Usage |
|-----------|--------|-------|
| **Développement** | GitHub Copilot, ChatGPT, Claude Sonnet | Aide au débogage, optimisation code, structuration README, génération dashboard Streamlit |
| **Frameworks ML/DL** | scikit-learn, TensorFlow/Keras 3, NumPy, Pandas | Implémentation modèles, entraînement, évaluation |
| **Visualisation** | Matplotlib, Seaborn, Plotly | Génération graphiques EDA, métriques, dashboard interactif |
| **Déploiement** | Streamlit, h5py | Interface utilisateur, chargement modèle H5 |

### Limites rencontrées
- **Déséquilibre extrême** : 74× entre la classe `nv` (5403 images) et `df` (73 images) → nécessite `class_weight` et augmentation de données
- **Recall mélanome** : 64% → insuffisant pour un usage clinique réel (cible >95% pour éviter faux négatifs)
- **Temps d'entraînement** : Transfer Learning ~30 min sur T4 GPU
- **Compatibilité** : Migration Keras 3 → TF 2.15 nécessite patch manuel du fichier H5 (suppression `quantization_config`)
- **Généralisation** : Modèle entraîné uniquement sur images dermatoscopiques HAM10000 (pas de photos smartphone)

---

## Reproductibilité
- Python 3.10+
- `requirements.txt` fourni avec versions pinées
- `random_state=42` sur tous les splits et modèles
- Seed NumPy fixé (`np.random.seed(42)`)
- Google Colab : High RAM CPU (ML) / T4 GPU (DL)
