# Skin Lesion Classification

## Dataset Overview
For this study, the open-source **Skin Cancer MNIST: HAM 10000 (Human Against Machine)** dataset will be used. It contains 10,000 dermoscopic images of pigmented skin lesions, with seven types of skin lesions namely: Basal cell carcinoma (BCC), actinic keratoses (AKIEC), benign keratosis (BKL), skin fibroma (DF), melanocytic nevus (NV), melanomas (MEL), and vascular skin lesion (VASC). All the images are in `.jpg` format, typically in colour (RGB), which represents the visual appearance of the skin lesions.

Dataset Link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T


## Steps Taken So Far

1. **Dataset Loading and Preprocessing**:
   - The dataset's metadata file (`HAM10000_metadata.csv`) is used to load information about each image.
   - The image paths are mapped from the metadata, allowing us to locate each image in the file system.
   - Each image is read using the `PIL` library and stored in the `skin_df` DataFrame.

2. **Image Visualization**:
   - Sampled **5 images** from each category (`dx`) and display them in a grid.
   - Images are displayed at their original resolution without resizing to maintain their quality.
   - **Matplotlib** is used to create subplots and visualize the sampled images, and we save the final plot at **DPI 500** to ensure high-quality output.

     
3. **Data Cleaning**
   - **Handling Missing Age Values**: The dataset has missing values for the `'age'` feature. These missing values are filled with the mean age of all patients to maintain consistency and avoid dropping rows with missing data.

   
5. **Data Visualization**
This section visualizes key features of the **HAM10000** dataset to understand the distribution of skin lesions:

- **Skin Lesion Types (`'dx'`)**: A bar plot showing the distribution of different lesion types like BCC, AKIEC, BKL, etc.
- **Lesion Localization (`'localization'`)**: A bar plot showing where skin lesions are most commonly located on the body.
- **Sex Distribution (`'sex'`)**: A bar plot showing the gender distribution of patients with skin lesions.
- **Age Distribution (`'age'`)**: A histogram visualizing the age range of patients affected by skin lesions.

These visualizations help to better understand the dataset’s structure and guide further analysis


## 6. **Model Training**

This section describes the deep learning models used for skin lesion classification:

- **ResNet50**: A deep convolutional neural network with 50 layers and residual connections. It helps in training very deep networks by solving the vanishing gradient problem and improves model accuracy.

- **DenseNet121**: A densely connected CNN with 121 layers. Each layer connects to every other layer, which promotes feature reuse, improves gradient flow, and enhances learning efficiency.

- **ALBEF (Align Before Fuse)**: A multimodal model that combines image features with text information using a vision transformer and BERT. It aligns image and text embeddings before fusing them, allowing for better classification by including metadata such as `dx_type` and `localization`.

![flowdiagram](https://github.com/user-attachments/assets/0bd792fa-1991-4ac8-b123-3bc4504ab0ae)

### 📌 Model Training Workflow

- **Data Split**: The dataset is divided into training, validation, and test sets using an 80-10-10 ratio with `train_test_split`.
- **Training the Models**: All models are trained using the `fit` method with early stopping to avoid overfitting.
- **Code Location**: All model training scripts are located in the `code` folder:
  - `RESENET50-Final.py` — ResNet50 model training
  - `DenseNet121-Final.py` — DenseNet121 model training
  - `ALBEF_Model_Code-Final.py` — ALBEF multimodal model training

---

## 7. **ALBEF Plots & Evaluation**

To visualize the performance of the ALBEF model, the following file is used:

- `ALBEF_Model_Curve_.py` — Generates accuracy and loss plots for training and validation.

---

## 8. **Data Loading and Exploration**

Before training, the dataset is loaded, cleaned, and explored:

- **`New-Code-dataLoading`** — Loads raw image and metadata files and creates a structured CSV file.
- **`Data Exploration.ipynb`** — A notebook for analyzing the dataset, plotting class distribution, and exploring metadata.

🗂️ **All code files are available in the `code` folder** for easy access and execution.


- 
