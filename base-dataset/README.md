# Base Dataset Description

This file contains the information about base dataset source.

## Base Dataset

We used the **[Real-Time Anomaly Detection in CCTV Surveillance
](https://www.kaggle.com/datasets/webadvisor/real-time-anomaly-detection-in-cctv-surveillance)** dataset from Kaggle.
The dataset contains 14 classes of anomalies in the form of video clips. From the dataset card:
> UCF Crime Dataset in the most suitable structure. Contains 1900 videos from 13 different categories. To ensure the
> quality of this dataset, it is trained ten annotators (having different levels of computer vision expertise) to collect
> the dataset. Using videos search on YouTube and LiveLeak using text search queries (with slight variations e.g. “car
> crash”, “road accident”) of each anomaly.

### Dataset Directory Structure

The dataset is structured as follows:
```
Real-Time Anomaly Detection in CCTV Surveillance
├── /data
│   ├── /abuse
│   ├── /arrest
│   ├── ... (other classes)
│   ├── /vandalism
│   ├── test.csv
│   └── train.csv

* Video format is : *.mp4
```
### Classes of the base dataset
- `abuse` (50 videos)
- `arrest`  (50 videos)
- `arson` (50 videos)
- `assault` (50 videos)
- `burglary` (100 videos)
- `explosion` (50 videos)
- `fighting` (50 videos)
- `normal` (950 videos)
- `roadaccident` (150 videos)
- `robbery` (150 videos)
- `shooting` (50 videos)
- `shoplifting` (50 videos)
- `stealing` (100 videos)
- `vandalism` (50 videos)

**Total Videos: 1900**

---

### Modifications

We modified this dataset for our project. 
See [`/dataset`](https://github.com/tinykishore/fydp-experiments/blob/master/dataset/README.md) for more information