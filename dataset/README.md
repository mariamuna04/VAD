# Dataset 

This file contains the description of the dataset used in the project.

## Dataset

The UCF-Crime dataset is a valuable resource for researchers developing anomaly detection algorithms for real-world
surveillance applications. This large-scale dataset comprises 1,900 untrimmed videos captured by surveillance cameras.
It encompasses a diverse range of 13 critical anomaly classes, including abuse, arson, assault, road accidents,
burglary, explosions, fighting, robbery, shooting, stealing, and vandalism. These events pose significant threats to
public safety and require prompt detection.

In the base dataset, we performed some pre-processing to create a new dataset that is suitable for our project. The
preprocessing steps are as follows:

- Discard irrelevant classes
- Video Segmentation
- Manual Labeling (Both binary and multi-class labels)
- Data Augmentation

### Discard irrelevant classes

The dataset additionally includes 950 videos showcasing normal activities within the surveillance environment. It's
important to note that the "arrest" and "shoplifting" classes were excluded due to their potential ambiguity for anomaly
detection. Apprehending criminals (arrests) is a necessary security measure, not an anomaly. Similarly, shoplifting
often involves subtle actions that may vary depending on the context, making it a challenging class for models to learn
definitively.

> **TL;DR:** We removed the "arrest" and "shoplifting" classes from the dataset.

### Video Segmentation

The UCF-Crime dataset videos are quite long, and the actual incidents we're looking for (like abuse or assault) only
happen for a brief moment. To tackle this, we cleverly chopped the videos into smaller, 10 seconds segment pieces. This
way, we wouldn't miss any crucial moments. We also removed any repetitive or unnecessary parts. In our opinion, 10
seconds is a reasonable duration for a model to learn the features of an anomaly, but this can be adjusted as needed.

> **TL;DR:** We split the videos into 10-second segments to focus on the critical moments.


### Manual Labeling

We observed every 10 seconds segment and manually labeled them as either anomaly classes or "normal". This process was
repeated for all the videos in the dataset. If a video segment contained an anomaly, we labeled it with the corresponding
anomaly class. Otherwise, we labeled it as "normal".

#### Binary Labeling

Initially, we labeled the segments as either "anomaly" or "normal". This binary labeling was useful for training models
to distinguish between normal and anomalous activities. In binary label the classes are balanced, equal number of
anomalous and normal videos.