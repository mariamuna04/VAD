## Dataset

---

This directory contains 2 subdirectories:

- `dataset/anomaly`
- `dataset/nonanomaly`

> This structure should be maintained because the code is written to read the dataset from these directories.

The `anomaly` directory contains the **10-second** segmented video clips from all 13 classes that are manually labeled as anomalies. The `nonanomaly` directory contains the 10-second segmented video clips that are manually labeled as non-anomalies.

`anomaly` class contains the following classes:
- [x] abuse
- [ ] <strike>arrest</strike> (Crossed out because arrest is not considered as an anomaly)
- [x] arson
- [x] assault
- [ ] burglary
- [x] explosion
- [x] fighting
- [ ] normal
- [x] roadaccident
- [x] robbery
- [x] shooting
- [ ] shoplifting
- [ ] stealing
- [x] vandalism

**Total Videos:** `1944`

`nonanomaly` class contains the following classes:
- normal
- arson
- abuse

**Total Videos:** `1711`

---
[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1-X32CoRPdcicm_wMTs43KB-653d641FY?usp=share_link)