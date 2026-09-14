"""
Fonctions de base pour la détection de visages, le calcul d'embeddings,
et la comparaison de similarité.
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# --- Initialisation du modèle (une seule fois, au chargement du module) ---
face_analysis = FaceAnalysis(name="buffalo_l")
face_analysis.prepare(ctx_id=-1, det_size=(320, 320))

# --- Cache pour éviter de recalculer les mêmes embeddings ---
_embedding_cache = {}


def detect_face(image_path, img_size=250):
    """
    Charge une image, détecte le(s) visage(s), et retourne le visage principal
    (le plus proche du centre de l'image si plusieurs sont détectés).

    Retourne : (img_rgb, face) ou (img_rgb, None) si aucun visage détecté.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_analysis.get(img_rgb)

    if len(faces) == 0:
        return img_rgb, None

    if len(faces) == 1:
        return img_rgb, faces[0]

    # plusieurs visages détectés → on prend celui le plus proche du centre
    center = img_size / 2

    def distance_to_center(face):
        x1, y1, x2, y2 = face.bbox
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        return (face_center_x - center) ** 2 + (face_center_y - center) ** 2

    best_face = min(faces, key=distance_to_center)
    return img_rgb, best_face


def get_embedding(image_path):
    """Retourne l'embedding normalisé du visage principal, ou None si non détecté."""
    _, face = detect_face(image_path)
    if face is None:
        return None
    return face.normed_embedding


def get_embedding_cached(image_path):
    """Version avec cache de get_embedding, pour éviter les recalculs inutiles."""
    if image_path in _embedding_cache:
        return _embedding_cache[image_path]
    embedding = get_embedding(image_path)
    _embedding_cache[image_path] = embedding
    return embedding


def cosine_similarity(embedding1, embedding2):
    """Similarité cosinus entre 2 embeddings normalisés (produit scalaire)."""
    return float(np.dot(embedding1, embedding2))


def compare_faces_cached(image_path1, image_path2):
    """
    Compare deux visages et retourne leur score de similarité.
    Retourne None si un visage n'a pas pu être détecté sur l'une des images.
    """
    embedding1 = get_embedding_cached(image_path1)
    embedding2 = get_embedding_cached(image_path2)

    if embedding1 is None or embedding2 is None:
        return None

    return cosine_similarity(embedding1, embedding2)


def verify(image_path1, image_path2, threshold=0.238):
    """
    Vérifie si deux images représentent la même personne.
    Retourne : (bool ou None, score ou None)
    """
    score = compare_faces_cached(image_path1, image_path2)
    if score is None:
        return None, None
    return score >= threshold, score