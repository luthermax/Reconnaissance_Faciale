"""
Génère les paires positives (même personne) et négatives (personnes différentes)
à partir du dataset LFW.

Structure attendue du dataset :
    data/raw/lfw/
        Nom_Personne_1/
            Nom_Personne_1_0001.jpg
            Nom_Personne_1_0002.jpg
        Nom_Personne_2/
            Nom_Personne_2_0001.jpg
        ...
"""

import os
import random
import csv
from itertools import combinations
from pathlib import Path

# ---------- Configuration ----------
DATASET_DIR = "data/raw/lfw_funneled"          # dossier contenant les sous-dossiers par personne
OUTPUT_DIR = "data/pairs"
MIN_PHOTOS_FOR_POSITIVE = 2            # nb minimum de photos pour créer une paire positive
MAX_POSITIVE_PAIRS_PER_PERSON = 3      # évite l'explosion combinatoire pour les gens très photographiés
N_NEGATIVE_PAIRS = None                # si None, sera égal au nombre de paires positives générées
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


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


def generate_positive_pairs(people, min_photos=2, max_pairs_per_person=3):
    """Génère des paires (photo1, photo2) de la même personne."""
    positive_pairs = []

    for name, photos in people.items():
        if len(photos) < min_photos:
            continue

        # toutes les combinaisons possibles de 2 photos pour cette personne
        possible_pairs = list(combinations(photos, 2))

        # limite le nombre de paires par personne pour équilibrer le dataset
        if len(possible_pairs) > max_pairs_per_person:
            possible_pairs = random.sample(possible_pairs, max_pairs_per_person)

        for img1, img2 in possible_pairs:
            positive_pairs.append((img1, img2))

    return positive_pairs


def generate_negative_pairs(people, n_pairs):
    """Génère des paires (photo1, photo2) de personnes différentes."""
    names = list(people.keys())
    negative_pairs = []
    seen = set()

    attempts = 0
    max_attempts = n_pairs * 20  # sécurité anti-boucle infinie

    while len(negative_pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        name1, name2 = random.sample(names, 2)

        # évite de générer deux fois la même paire de personnes
        pair_key = tuple(sorted([name1, name2]))
        if pair_key in seen:
            continue

        img1 = random.choice(people[name1])
        img2 = random.choice(people[name2])

        negative_pairs.append((img1, img2))
        seen.add(pair_key)

    return negative_pairs


def save_pairs_to_csv(pairs, output_path):
    """Sauvegarde une liste de paires (img1, img2) dans un fichier CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image1", "image2"])
        writer.writerows(pairs)

    print(f"{len(pairs)} paires sauvegardées dans {output_path}")


def main():
    print("Chargement des personnes et de leurs photos...")
    people = list_people_with_photos(DATASET_DIR)
    print(f"{len(people)} personnes trouvées dans le dataset.")

    eligible = {n: p for n, p in people.items() if len(p) >= MIN_PHOTOS_FOR_POSITIVE}
    print(f"{len(eligible)} personnes ont au moins {MIN_PHOTOS_FOR_POSITIVE} photos.")

    # --- Paires positives ---
    positive_pairs = generate_positive_pairs(
        eligible,
        min_photos=MIN_PHOTOS_FOR_POSITIVE,
        max_pairs_per_person=MAX_POSITIVE_PAIRS_PER_PERSON,
    )
    save_pairs_to_csv(positive_pairs, os.path.join(OUTPUT_DIR, "positive_pairs.csv"))

    # --- Paires négatives ---
    n_negative = N_NEGATIVE_PAIRS or len(positive_pairs)
    negative_pairs = generate_negative_pairs(people, n_negative)
    save_pairs_to_csv(negative_pairs, os.path.join(OUTPUT_DIR, "negative_pairs.csv"))

    print("\nTerminé.")
    print(f"  - Paires positives : {len(positive_pairs)}")
    print(f"  - Paires négatives : {len(negative_pairs)}")


if __name__ == "__main__":
    main()
