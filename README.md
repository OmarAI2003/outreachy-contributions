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

### Data Limitations:
1) Data Size (404 compounds):
    - Predictions are as good as the training data. Small dataset may miss chemical diversity, leading to poor predictions for structurally distinct compounds.
2)  [Applicability Domain (AD):](https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/applicability-domain)
    - Models trained on this data may fail for molecules  with structures too different from the training data (e.g., new types of atoms/bonds), requiring AD checks("warning" to spot nonsimilar cases) to flag unreliable predictions.
### Dataset License: 
- [CC BY 4.0.](https://creativecommons.org/licenses/by/4.0/)

#### Concepts & Abbreviations:
- **susceptible individuals:** Refer to those who are genetically or biologically predisposed to developing allergic reactions upon repeated exposure to chemical agents

- **Applicability Domain (AD)**: Chemical space where the model's predictions are reliable. Domain is determined by Model descriptors and responses.

### **Why This Dataset?**
I chosed this dataset because it connects to everyday life. Think of reactions to cosmetics, detergents, or skincare products. Skin issues are universal. This dataset helps predict chemical safety without relying heavily on animal testing, aligning with my interest in ethical alternatives. It’s also a classifier (safe vs. unsafe) and the models built from it could be easier to interpret. The size dataset is manageable for my computational setup. By working on this, I’m tackling a real-world problem that blends health, consumer products, and sustainability which feels both meaningful.
__________________