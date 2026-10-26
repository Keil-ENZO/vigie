# Validation AIS du radar Sentinel-1, CFAR log-normal VV et VH

## Passes du même jour que l'optique, non vues au réglage

| Zone | Passe S1, UTC | AIS | AIS > 50 m | Détections | Appariées | Sombres | Précision | Rappel > 50 m | Rappel > 50 m eau libre | Longueur, err. méd. |
|---|---|---|---|---|---|---|---|---|---|---|
| Skagen, mouillage | 2026-10-18 17:01 | 74 | 27 | 21 | 21 | 0 | 100 % | 78 % | 19/20 | 12.0 m |
| Aarhus, baie | 2026-10-18 17:00 | 28 | 8 | 13 | 8 | 5 | 62 % | 50 % | 1/1 | 73.0 m |
| Fredericia, Petit Belt | 2026-10-18 17:00 | 11 | 2 | 4 | 4 | 0 | 100 % | 50 % | 1/2 | 101.0 m |
| Øresund, Copenhague | 2026-09-27 05:24 | 123 | 7 | 33 | 15 | 18 | 46 % | 100 % | 5/5 | 40.0 m |
| Grand Belt, pont | 2026-09-30 17:00 | 5 | 4 | 9 | 4 | 5 | 44 % | 100 % | 4/4 | 20.5 m |
| **Total** | | 241 | 48 | 80 | 52 | 28 | **65 %** | **77 %** | **94 %** | 24.5 m, biais 10.5 m |
| *Optique S2, mêmes zones et jours* | | 262 | 47 | 95 | 47 | 48 | 50 % | 51 % | – | 19.0 m |

## Passes de réglage des seuils

| Zone | Passe S1, UTC | AIS | AIS > 50 m | Détections | Appariées | Sombres | Précision | Rappel > 50 m | Rappel > 50 m eau libre | Longueur, err. méd. |
|---|---|---|---|---|---|---|---|---|---|---|
| Skagen, mouillage | 2026-09-30 17:01 | 82 | 24 | 20 | 18 | 2 | 90 % | 71 % | 13/13 | 25.5 m |
| Øresund, Copenhague | 2026-09-30 17:00 | 126 | 7 | 21 | 16 | 5 | 76 % | 86 % | 4/4 | 22.5 m |
| **Total** | | 208 | 31 | 41 | 34 | 7 | **83 %** | **74 %** | **100 %** | 22.5 m, biais 3.0 m |

Écart médian à l'AIS : 43 m à l'arrêt (63 navires), 55 m en route à 3 nœuds ou plus (21 navires).
