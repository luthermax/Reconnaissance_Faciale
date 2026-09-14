# Vérification d'identité par visage

Projet personnel de reconnaissance faciale couvrant à la fois la **vérification 1:1**
(deux visages sont-ils la même personne ?) et l'**identification 1:N** (à qui appartient
ce visage, parmi une base de personnes enrôlées ?), avec une API pour rendre le système
utilisable.

## Cas d'usage

Contrôle d'accès simple : seules les personnes enrôlées dans une base restreinte
peuvent être authentifiées (ex. accès à une salle, un compte, un service).

## Dataset

[LFW — Labeled Faces in the Wild](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
(~13 000 images, ~5 700 personnes). Utilisé filtré aux personnes ayant au moins 2 photos,
pour pouvoir constituer des paires de test.

## Pipeline technique

```
Image → Détection de visage → Alignement → Embedding (512D) → Comparaison / Recherche
```

- **Détection + embedding** : [InsightFace](https://github.com/deepinsight/insightface)
  (modèle `buffalo_l`, détecteur RetinaFace + encodeur ArcFace)
- **Recherche 1:N** : [FAISS](https://github.com/facebookresearch/faiss)
  (`IndexFlatIP`, recherche par produit scalaire sur embeddings normalisés)
- **API** : FastAPI, endpoints `/verify` (1:1), `/identify` (1:N) et `/enroll` (ajout
  dynamique d'une personne à la base)

## Choix techniques et difficultés rencontrées

- **Images multi-visages** : certaines photos LFW contiennent plusieurs visages
  (personnes en arrière-plan). Règle retenue : garder le visage dont le centre de la
  bounding box est le plus proche du centre de l'image — cohérent avec le fait que LFW
  cadre toujours la personne nommée comme sujet principal.
- **Performance** : la taille de détection par défaut d'InsightFace (640×640) est
  surdimensionnée pour des images LFW de 250×250, et le recalcul répété des mêmes
  embeddings (une image pouvant apparaître dans plusieurs paires) ralentissait fortement
  le traitement. Corrigé en réduisant `det_size` à 320×320 et en ajoutant un cache
  d'embeddings par chemin d'image.
- **Choix du seuil de décision** : plutôt qu'un seuil arbitraire, le seuil a été
  déterminé via une courbe ROC sur l'ensemble des paires positives/négatives générées,
  en maximisant l'indice de Youden (TPR − FPR).
- **Persistance de la base** : l'index FAISS et les labels associés sont sauvegardés sur
  disque (`database/embeddings.index`, `database/labels.json`) et rechargés au démarrage
  de l'API, plutôt que recalculés à chaque redémarrage.

## Résultats

### Vérification 1:1

| Métrique | Valeur |
|---|---|
| AUC (courbe ROC) | 0.9995 |
| Seuil retenu | 0.238 |
| Similarité moyenne — paires positives | 0.65 (min 0.015, max 0.97) |
| Similarité moyenne — paires négatives | 0.006 (min -0.18, max 0.23) |

Les deux distributions de scores sont très nettement séparées, avec un chevauchement
quasi inexistant.

### Identification 1:N

Base de test : 30 personnes enrôlées.

| Métrique | Valeur |
|---|---|
| Taux d'identification correcte | 100 % |
| Taux de rejet correct (personnes non enrôlées) | 100 % |
| Taux de fausse acceptation | 0 % |

## Limites et pistes d'amélioration

- **Base de test réduite** (30 personnes) : un taux de fausse acceptation de 0 % est
  plus facile à obtenir sur un petit nombre de personnes que sur une base de plusieurs
  milliers, où le risque de confusion entre visages similaires augmente.
- **Dataset favorable** : LFW contient des photos globalement bien cadrées et de bonne
  qualité. En conditions réelles (webcam, éclairage variable, angles de prise de vue
  difficiles), les performances seraient probablement inférieures à celles mesurées ici.
- **Pas de détection d'usurpation (anti-spoofing)** : le système ne vérifie pas si le
  visage présenté est une vraie personne ou une photo/vidéo — piste d'amélioration pour
  un usage en sécurité réelle.
- **Seuil unique** : le seuil de 0.238 a été calibré sur LFW ; il devrait être recalculé
  si le système est utilisé dans d'autres conditions (photos prises autrement).

## Architecture du projet

```
face-id-verification/
├── app/
│   ├── face_utils.py     # détection, embedding, comparaison
│   ├── database.py       # index FAISS, persistance, identification 1:N, enrôlement
│   └── main.py           # API FastAPI (verify, identify, enroll)
├── data/
│   ├── raw/               # dataset LFW (non versionné, voir lien ci-dessus)
│   └── pairs/              # paires positives / négatives (CSV)
├── database/               # index FAISS sauvegardé (non versionné, généré au 1er lancement)
├── generate_pairs.py      # script de génération des paires
├── notebook.ipynb         # exploration, tests, courbe ROC, évaluation
├── requirements.txt
├── .gitignore
└── README.md
```

## Lancer le projet

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Interface de test : `http://127.0.0.1:8000/docs`

### Endpoints disponibles

- `POST /verify` — envoyer deux images, retourne `match` (bool) et `score`
- `POST /identify` — envoyer une image, retourne le nom identifié (ou `"Inconnu"`) et le score
- `POST /enroll` — envoyer un `name` et une image, ajoute la personne à la base
