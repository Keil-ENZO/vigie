# Validation AIS, modèle `defendintel-yolov8s-s2.pt`, seuil 0.2

| Zone | Passe | AIS | AIS > 50 m | Détections | Appariées | Sombres | Précision | Rappel > 50 m | Longueur, err. méd. |
|---|---|---|---|---|---|---|---|---|---|
| Skagen, mouillage | S2B_32VNJ_20261018_0_L2A | 66 | 29 | 23 | 23 | 0 | 100 % | 76 % | 30.0 m |
| Aarhus, baie | S2B_32VNH_20261018_0_L2A | 28 | 9 | 2 | 1 | 1 | 50 % | 11 % | 43.0 m |
| Fredericia, Petit Belt | S2B_32UNG_20261018_0_L2A | 15 | 2 | 11 | 1 | 10 | 9 % | 0 % | 8.0 m |
| Øresund, Copenhague | S2C_33UUB_20260927_0_L2A | 147 | 6 | 50 | 20 | 30 | 40 % | 0 % | 18.5 m |
| Grand Belt, pont | S2C_32UPG_20260930_0_L2A | 6 | 1 | 9 | 2 | 7 | 22 % | 100 % | 6.5 m |
| **Total** | | 262 | 47 | 95 | 47 | 48 | **50 %** | **51 %** | 19.0 m, biais 5.0 m |
