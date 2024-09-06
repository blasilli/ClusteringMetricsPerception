# Towards a Visual Perception-Based Analysis of Clustering Quality Metrics
> Graziano Blasilli, Daniel Kerrigan, Enrico Bertini, and Giuseppe Santucci.

## Abstract
Clustering is an essential technique across various domains, such as data science, machine learning, and explainable artificial intelligence. 
Information visualization and visual analytics techniques have been proven to effectively support human involvement in the visual exploration of clustered data to enhance the understanding and refinement of cluster assignments. To support the human involvement, several perceptual studies and visual quality metrics have already been proposed. However, the visual perception of clustering quality metrics, also known as Cluster Validity Indexes (CVIs), still remains to be further explored.
This paper presents the first attempt of a deep and exhaustive evaluation of the perceptive aspects of clustering quality metrics, focusing on the Davies-Bouldin Index, Dunn Index, Calinski-Harabasz Index, and Silhouette Score.
Our research is centered around two main objectives: a) assessing the human perception of common CVIs in 2D scatterplots and b) exploring the potential of Large Multimodal Models, in particular GPT-4o, to emulate the assessed human perception.
To this end, we conducted two systematic data studies and a user study covering a broad collection of datasets. By discussing the obtained results, highlighting limitations, and areas for further exploration, this paper aims to propose a foundation for future research activities.

## Preprint Paper
[[Preprint] Towards a Visual Perception-Based Analysis of Clustering Quality Metrics.pdf](https://github.com/blasilli/ClusteringMetricsPerception/blob/main/%5BPreprint%5D%20Towards%20a%20Visual%20Perception-Based%20Analysis%20of%20Clustering%20Quality%20Metrics.pdf)

## Data Collections
This repository contains all the materials used in the paper, including datasets, user study responses, charts, and estimations related to the perception of clustering validity indexes (CVIs).

### SDPC
- `csv/`: Contains CSV files for the SDPC-72 datasets.
- `img-pdf/`: Contains PDF images of individual scatterplots.
- `img-png/`: Contains PNG images of individual scatterplots.
- `imgpairs-pdf/`: Contains PDF images of scatterplot pairs.
- `imgpairs-png/`: Contains PNG images of scatterplot pairs.
- `datasets.csv`: Contains metadata about the datasets, such as metric values.
- `pairs.csv`: Contains information about the pairs, such as RPD values.

### SDPC-72
- `csv/`: Contains CSV files for the SDPC-72 datasets.
- `img-pdf/`: Contains PDF images of individual scatterplots.
- `img-png/`: Contains PNG images of individual scatterplots.
- `imgpairs-pdf/`: Contains PDF images of scatterplot pairs.
- `imgpairs-png/`: Contains PNG images of scatterplot pairs.
- `datasets.csv`: Contains metadata about the datasets, such as metric values.
- `pairs.csv`: Contains information about the pairs, such as RPD values.



## Experiments
### Experiment-1
#### Experiment-1.1
- `charts/`: Contains resulting charts generated during Experiment 1.1.
- `estimation/`: Contains probability estimation results.
- `raw/`: Contains raw data, generated from user responses, used in Experiment 1.1.
- `responses-users.csv`: Contains user responses collected during Experiment 1.1.

#### Experiment-1.2
- `charts/`: Contains resulting charts generated during Experiment 1.2.
- `estimation/`: Contains probability estimation results.
- `raw/`: Contains raw data, generated from user responses, used in Experiment 1.2.
- `responses-users.csv`: Contains user responses collected during Experiment 1.2.

#### Experiment-1-Overall
- `charts/`: Contains resulting charts generated as overall results of Experiment 1.
- `estimation/`: Contains probability estimations generated as overall results of Experiment 1.

### Experiment-2
- `charts/`: Contains resulting charts generated during Experiment 2.
- `estimation/`: Contains probability estimation results.
- `raw/`: Contains raw data, generated from responses, used in Experiment 2.
- `responses-gpt4o.csv`: Contains responses from GPT-4o during Experiment 2.



