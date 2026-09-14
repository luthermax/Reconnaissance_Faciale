"""
API de vérification d'identité par visage.

Lancer avec :
    uvicorn app.main:app --reload

Documentation interactive disponible sur :
    http://127.0.0.1:8000/docs
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form

from app.face_utils import verify as verify_faces
from app.database import face_db

app = FastAPI(title="Face ID Verification API")


@app.on_event("startup")
def startup_event():
    """Charge (ou construit) la base de personnes enrôlées au démarrage de l'API."""
    n = face_db.build()
    print(f"Base prête : {n} personnes enrôlées.")


def _save_upload_to_temp(upload_file: UploadFile) -> str:
    """Sauvegarde un fichier uploadé dans un fichier temporaire et retourne son chemin."""
    suffix = Path(upload_file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
        return tmp.name


def _cleanup(*paths):
    """Supprime les fichiers temporaires après traitement."""
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


@app.post("/verify")
async def verify_endpoint(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    """
    Vérifie si deux images uploadées représentent la même personne (1:1).
    """
    path1 = _save_upload_to_temp(image1)
    path2 = _save_upload_to_temp(image2)

    try:
        match, score = verify_faces(path1, path2)
    finally:
        _cleanup(path1, path2)

    if match is None:
        return {"error": "Visage non détecté sur au moins une des deux images."}

    return {"match": match, "score": round(score, 4)}


@app.post("/identify")
async def identify_endpoint(image: UploadFile = File(...)):
    """
    Identifie une personne parmi la base enrôlée (1:N).
    """
    path = _save_upload_to_temp(image)

    try:
        name, score = face_db.identify(path)
    finally:
        _cleanup(path)

    if name is None:
        return {"error": "Visage non détecté sur l'image."}

    return {"name": name, "score": round(score, 4)}


@app.post("/enroll")
async def enroll_endpoint(name: str = Form(...), image: UploadFile = File(...)):
    """
    Ajoute une nouvelle personne à la base enrôlée.
    """
    path = _save_upload_to_temp(image)

    try:
        success, message = face_db.enroll(name, path)
    finally:
        _cleanup(path)

    if not success:
        return {"error": message}

    return {"message": message}


@app.get("/")
def root():
    return {"status": "ok", "enrolled_count": len(face_db.labels)}