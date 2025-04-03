## Table of Contents
- [Dataset](#dataset-chemically-induced-skin-reactions)
- [Exploratory Data Analysis (EDA)](exploratory-data-analysis-(eda))
- [Selecting Featurisers](Featurizer-Selection)


## Dataset: [Chemically-induced Skin Reactions](https://tdcommons.ai/single_pred_tasks/tox#skin-reaction)

### Dataset Overview
The data endpoint is to identify skin sensitizers => substances that can cause allergic [contact dermatitis (ACD)](https://ntp.niehs.nih.gov/whatwestudy/niceatm/test-method-evaluations/skin-sens) in humans through an immune response triggered by attatching chemicals to [skin proteins](https://ntp.niehs.nih.gov/whatwestudy/niceatm/test-method-evaluations/skin-sens/easa/easa) resulting from repeated exposure in susceptible individuals. The dataset is set for binary **classification** tasks, where each drug is represented by its SMILES string and labeled as either causing a skin reaction (1) or not (0). Allergic skin reactions can appear in the form of redness, swelling, or itching in humans. The data collected results from tests to facilitate the evaluation of alternative methods to predict reactions without using humans or animals.


### Data Source
Data was collected from non-animal defined approaches (DAs). Chemica tests like ([DPRA](https://ntp.niehs.nih.gov/whatwestudy/niceatm/test-method-evaluations/skin-sens/da)))=> measures how chemicals bind to skin proteins. Tests for skin cells that checks if chemicals trigger inflammation signals in human skin cells like [KeratinoSens](https://ntp.niehs.nih.gov/whatwestudy/niceatm/test-method-evaluations/skin-sens/da). Immune cells test [h-CLAT](https://ntp.niehs.nih.gov/whatwestudy/niceatm/test-method-evaluations/skin-sens/da) that detects chemicals that activate allergy-related immune cells. Then tests were grouped to a  workflow (OECD guidelines) and their results were:

- Compared to **past human skin data** ([HPPT](https://ntp.niehs.nih.gov/whatwestudy/niceatm/test-method-evaluations/skin-sens/hppt)) and **animal test data** ([LLNA](https://ntp.niehs.nih.gov/whatwestudy/niceatm/test-method-evaluations/skin-sens/llna)) to confirm accuracy.

-  Combined using **agreed-upon rules** (from [multi-lab studies](https://www.sciencedirect.com/science/article/abs/pii/S0041008X14004529)) to create final labels (1=allergen, 0=nott allergen).

### Dataset Size:
- 404 drugs in the form of SMILES strings.

### Dataset Split method (may get changed in Hyperparameter Tuning):
Scaffold groups compounds by core structure, ensuring the model is evaluated distinct chemicals. I also used it For consistency with domain best practices,
I set the seed to the good number 42 for reproducability.
I used 80/10/10 split to allocate more data for the training set thus giving enough data for model learning.
### Data Limitations:
1) Data Size (404 compounds):
    - Predictions are as good as the training data. Small dataset may miss chemical diversity, leading to poor predictions for structurally distinct compounds.
2)  [Applicability Domain (AD):](https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/applicability-domain)
    - Models trained on this data may fail for molecules with structures too different from the training data (e.g., new types of atoms/bonds), requiring AD checks("warning" to spot nonsimilar cases) to flag unreliable predictions.
### Dataset License:
- [CC BY 4.0.](https://creativecommons.org/licenses/by/4.0/)

#### Concepts & Abbreviations:
- **susceptible individuals:** Refer to those who are genetically predisposed to developing allergic reactions upon repeated exposure to chemical agents

- **Applicability Domain (AD)**: Chemical space where the model's predictions are reliable. Domain is determined by Model descriptors and responses.

### **Why This Dataset?**
I chosed this dataset because it connects to everyday life. Think of reactions to cosmetics, detergents, or skincare products. Skin issues are universal. This dataset helps predict chemical safety without relying heavily on animal testing, aligning with my interest in ethical alternatives. It’s also a classifier (safe vs. unsafe) and the models built from it could be easier to interpret. The size dataset is manageable for my computational setup. By working on this, I’m tackling a real-world problem that blends health, consumer products, and sustainability which feels both meaningful.
__________________

## Exploratory Data Analysis (EDA)

Although direct analysis on raw SMILES strings is hard. I did some EDA to prepare for featurization and to identify issues or biases in the data that could impact model performance.



Initial exploration to understand the dataset's basic structure & content.

### 1. Dataset Dimensions

The dataset has **404 entries (rows)**, each representing a unique chemical compound, and **3 features (columns)**.

### 2. Data Columns and Types:

A summary of the dataset columns, their data types, and non-null counts is provided below:

| Column  | Non-Null Count | Dtype  | Description                           |
|---------|----------------|--------|---------------------------------------|
| Drug_ID | 404            | string | Identifier for the chemical compound  |
| Drug    | 404            | string | chemical SMILES string.               |
| Y       | 404            | int64  | Target label (1: Reaction, 0: No Reaction) |

**Findings:**

* Analysis confirms that there are **no missing (null) values** in all 404 entries. This indicates good data and simplifies the initial preprocessing steps. no imputation is needed for missing values.

________________________
### 3. Data Uniqueness

I checked the uniquenes of data values for each column to assure no duplicates.

| Column  | Unique Values | Observation                                            |
|---------|---------------|--------------------------------------------------------|
| Drug_ID | 404           | All identifiers are unique.                            |
| Drug    | 404           | SMILES strings (representing molecules) are unique.    |
| Y       | 2             | Target has 2 distinct values (0 and 1).                |

**Key Findings:**

* 404 unique values in both the `Drug_ID` and `Drug` columns, matching the total number of rows (404), confirming that **each entry represents a unique chemical compound**. There are no duplicate molecules in the data.
* predictor variable `Y` has 2 unique values consistent with the **binary classification**.
____________________________________________

### 4. Class Distribution (Target Variable Balance)

The `Y` target variable's distribution , tells if a chemical causes a skin reaction.I checked it to see if it has imbalance.

* **Class 1 (Sensitizers - Causes Reaction):** `274` samples
* **Class 0 (Non-sensitizers - No Reaction):** 130 samples

approximates to:
* 67.8% Class 1 (Sensitizers)
* 32.2% Class 0 (Non-sensitizers)

The class distribution is visualized in the bar plot below:

![Class Balance Bar Plot](images/class_balance_bar.png)

*Figure 1: Distribution of samples across the two classes.*

**Key Finding:** There's a **class imbalance**. The positive class (1, Sensitizers) got more than twice as many samples (274) as the negative class (0, Non-sensitizers) (130). This imbalance (ratio ≈ 2.1:1) is significant and needs to be considered during modeling. Training a model on this data without accounting for imbalance can lead to a bias towards predicting the majority class (Sensitizers). I will try some resampling (SMOTE, over/under-sampling) or class weights during training. Evaluation metrics sensitive to imbalance, like the F1-score or Precision-Recall AUC.
_________________________

### 5. SMILES String Length Analysis

To get an idea of molecular size and complexity, I analyzed the length of SMILES strings in `Drug` column.

**SMILES string lengths escriptive stats:**:

| Statistic         | Value |
|-------------------|-------|
| Count             | 404   |
| Mean Length       | 25.4  |
| Standard Deviation| 15.9  |
| Minimum Length    | 4     |
| 25th Percentile   | 15.8  |
| Median (50%)      | 22.0  |
| 75th Percentile   | 30.0  |
| Maximum Length    | 98    |

**Distribution Insights:**

Histogram with summry stats to show length distribution:

![SMILES Length Distribution Plot](images/smiles_length_histogram.png)
*Fig 2: Distribution (Hist & KDE) of SMILES string lengths, showing Mean, Min, and Max.*

* **Typical Molecule Size:** The median length is 22 characters, and the histogram/KDE plot peaks between approximately 10 and 25 characters. This suggests significant portion is of relatively small molecules. 50% of the molecules have SMILES lengths between ~16 and 30 characters (IQR).
* **Variability and Skewness:** The distribution is **right-skewed**, as indicated by the mean (25.4) > than the median (22) and the long tail extending to higher lengths. This skewness highlights the presence of some large complex molecules.
* **Range:**: 4 To 98 is a wide range of lengths. Big Standard deviation (15.9 relative to the mean), shows great diversity in molecular size/complexity.

**Key Finding:** Data has lots of small-to-medium-sized molecules, but also includes a number of larger, maybe more complex structures. our Data is considerd hetrognieous.
_________________________________________________________

### Featurizer Selection


Starting my search journey,

I initially considered `[eos4avb](https://github.com/ersilia-os/eos4avb)` from Ersilia's hub, which claims readiness for **drug toxicity** prediction, but decided not to go with for these:

- **Different Endpoint**: Designed for broad bioactivity prediction for multiple domains, unlike my nuanced immune triggered skin sensitization.
- **Model Input Data**: Uses molecule images, may losing chemical interactions and details essential for predicting protein-chemical binding and inflammatory responses.
- **Dataset Limitations**: My small dataset requires precise feature extractor, while this general image appraoch may not capture raliable skin perdictions.

After a review of available featurizers, considered task requirements (predicting immune-triggered skin sensitization on a small, heterogeneous data).I identified Featurizers that aren't suitable for this Task like reaction transformers, format converters, text generators, translators and image-based models.


**I narrowed down to these 2 models:**

The two selected featurizers work in togeather to surpass challenges revealed by EDA on our small dataset like class imbalance, small sample size, and heterogeneity in molecule sizes.

1) **[Compound Embeddings](https://www.nature.com/articles/s41467-023-41512-2)** : use transfer learning that's good for my small dataset to generate 1024D vectors that integrate both physicochemical and bioactivity data. This model got knowledge from millions of compounds trained on [FS-Mol](https://www.microsoft.com/en-us/research/publication/fs-mol-a-few-shot-learning-dataset-of-molecules/?utm_source=chatgpt.com) and [ChEMBL](https://www.ebi.ac.uk/chembl/?utm_source=chatgpt.com) datasets mitigating my small dataset nad capturing complex relations like protein binding and immune response that causes skin sensitization.

2) **[Morgan Fingerprints](https://research.ibm.com/publications/morgangen-generative-modeling-of-smiles-using-morgan-fingerprint-features?utm_source=chatgpt.com)**: Most widely used molecular representation and outputs a 2048-dimensional binary vector, capturing structural features around atoms. This is important in skin reactions prediction, where specific reactive locations determine the outcome. it's already validated against so many toxicity studies [[comparision between molecular fingerprints and other descriptors ](https://pubs.acs.org/doi/10.1021/acs.chemrestox.0c00303?utm_source=chatgpt.com)] making it reliable basline. the binary representation of its 2048 features assures fast computation and easy integration to ML workflows


Together, these approaches offer a balanced view: the embeddings bring in a holistic, data-enriched perspective while Morgan Fingerprints guarantee the capture of fine-grained chemical details. This strategy is designed to achive model generalization accuracy, addressing the limitations and biases identified during EDA.
____________
### Featurisation Instructions:

The [`data_loader_and_featurizer.py`](scripts/data_loader_and_featurizer.py) takes your chemical data, split it into meaningful subsets model-ready, and generate featurised[`data files`](data/) to capture molecular information. Here’s how it works:

- **Data Preparation** – It obtains Skin Reaction dataset and divides it into three parts: training, validation, and test splits. These subsets are saved as separate files, each contains both feature data (the unique compound identifiers) and their labels.

- **Feature Generation** – Once data is split, the pipeline runs a featurisation step. where, a choosed fetched model (e.g:Morgan fingerprints based) processes the obtained input data files and set output file names.

- Output for Analysis – After processing, outputed new versions of dataset split files  are stored in the same [`data directory`](data/) for easy access.

Users need to run the [`data_loader and featurizer`](scripts/data_loader_and_featurizer.py) script to laod raw data from [TDC](https://tdcommons.ai/single_pred_tasks/tox#skin-reaction) then run the featurisation function within notebook by giving it a **fetched** model and file names.
