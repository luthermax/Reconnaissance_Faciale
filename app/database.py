"""
Construction de la base de personnes enrôlées (index FAISS), avec persistance
sur disque et ajout dynamique de nouvelles personnes.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from app.face_utils import get_embedding_cached

DATASET_DIR = "data/raw/lfw_funneled"  # adapte si besoin
MIN_PHOTOS = 2
N_ENROLLED = 30

INDEX_PATH = "database/embeddings.index"
LABELS_PATH = "database/labels.json"
EMBEDDING_DIM = 512


def list_people_with_photos(dataset_dir):
    """Retourne un dict {nom_personne: [chemins_photos]}."""
    people = {}
    dataset_path = Path(dataset_dir)
    for person_dir in dataset_path.iterdir():
        if person_dir.is_dir():
            photos = sorted([
                str(p) for p in person_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ])
            if photos:
                people[person_dir.name] = photos
    return people


class FaceDatabase:
    """Encapsule l'index FAISS et les labels associés, avec persistance sur disque."""

    def __init__(self):
        self.index = None
        self.labels = []
        self.eligible = {}

    def _save(self):
        Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)

    def _load(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

    def build(self, force_rebuild=False):
        """
        Charge la base depuis le disque si elle existe déjà, sinon la construit
        depuis le dataset LFW et la sauvegarde.
        """
        # on garde une trace de eligible pour les tests / l'évaluation dans le notebook
        people = list_people_with_photos(DATASET_DIR)
        self.eligible = {n: p for n, p in people.items() if len(p) >= MIN_PHOTOS}

        if not force_rebuild and Path(INDEX_PATH).exists() and Path(LABELS_PATH).exists():
            self._load()
            return len(self.labels)

        enrolled_names = list(self.eligible.keys())[:N_ENROLLED]

        embeddings = []
        labels = []
        for name in enrolled_names:
            ref_photo = self.eligible[name][0]
            embedding = get_embedding_cached(ref_photo)
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(name)

        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings)
        self.labels = labels

        self._save()
        return len(labels)

    def enroll(self, name, image_path):
        """
        Ajoute une nouvelle personne à la base à partir d'une photo, et sauvegarde
        l'index mis à jour sur disque.

        Retourne : (bool succès, message)
        """
        embedding = get_embedding_cached(image_path)
        if embedding is None:
            return False, "Aucun visage détecté sur l'image fournie."

        embedding = np.array([embedding]).astype("float32")
        self.index.add(embedding)
        self.labels.append(name)
        self._save()

        return True, f"{name} ajouté(e) à la base ({self.index.ntotal} personnes enrôlées)."

    def identify(self, image_path, threshold=0.238, k=1):
        """
        Cherche la personne la plus proche dans la base enrôlée.
        Retourne : (nom ou 'Inconnu' ou None, score ou None)
        """
        embedding = get_embedding_cached(image_path)
        if embedding is None:
            return None, None

        embedding = np.array([embedding]).astype("float32")
        scores, indices = self.index.search(embedding, k)

        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])

        if best_score < threshold:
            return "Inconnu", best_score

        return self.labels[best_idx], best_score


# Instance unique, construite (ou chargée) au démarrage de l'API
face_db = FaceDatabase()