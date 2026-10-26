"""Zones prêtes à scanner. Boîtes ouest, sud, est, nord, d'une dizaine de kilomètres."""

PRESETS = [
    {"id": "toulon", "name": "Rade de Toulon", "bbox": [5.86, 43.06, 5.98, 43.13],
     "why": "Première base navale française, ferries vers la Corse."},
    {"id": "marseille", "name": "Port de Marseille", "bbox": [5.28, 43.27, 5.40, 43.36],
     "why": "Bassins est, ferries et croisière."},
    {"id": "fos", "name": "Golfe de Fos", "bbox": [4.80, 43.33, 4.98, 43.45],
     "why": "Terminaux pétroliers et méthaniers, mouillages d'attente."},
    {"id": "brest", "name": "Rade de Brest", "bbox": [-4.56, 48.27, -4.40, 48.39],
     "why": "Base navale, sous-marins à l'Île Longue."},
    {"id": "lehavre", "name": "Le Havre", "bbox": [0.02, 49.43, 0.22, 49.52],
     "why": "Premier port à conteneurs français."},
    {"id": "skagen", "name": "Skagen, mouillage", "bbox": [10.50, 57.55, 10.80, 57.72],
     "why": "Grand mouillage à la pointe du Danemark. AIS ouvert de l'autorité danoise."},
    {"id": "rotterdam", "name": "Rotterdam, Maasvlakte", "bbox": [3.95, 51.90, 4.15, 52.02],
     "why": "Premier port d'Europe, porte-conteneurs géants."},
    {"id": "gibraltar", "name": "Détroit de Gibraltar", "bbox": [-5.50, 35.92, -5.25, 36.15],
     "why": "Rail de navigation entre Atlantique et Méditerranée."},
    {"id": "suez", "name": "Suez, entrée sud", "bbox": [32.45, 29.85, 32.65, 30.00],
     "why": "Navires en attente du convoi vers le nord."},
    {"id": "singapore", "name": "Détroit de Singapour", "bbox": [103.75, 1.15, 103.95, 1.28],
     "why": "Le mouillage le plus dense du monde."},
]
