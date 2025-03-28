# Skin Lesion Classification

## Dataset Overview
For this study, the open-source **Skin Cancer MNIST: HAM 10000 (Human Against Machine)** dataset will be used. It contains 10,000 dermoscopic images of pigmented skin lesions, with seven types of skin lesions namely: Basal cell carcinoma (BCC), actinic keratoses (AKIEC), benign keratosis (BKL), skin fibroma (DF), melanocytic nevus (NV), melanomas (MEL), and vascular skin lesion (VASC). All the images are in `.jpg` format, typically in colour (RGB), which represents the visual appearance of the skin lesions.

## Steps Taken So Far

1. **Dataset Loading and Preprocessing**:
   - The dataset's metadata file (`HAM10000_metadata.csv`) is used to load information about each image.
   - The image paths are mapped from the metadata, allowing us to locate each image in the file system.
   - Each image is read using the `PIL` library and stored in the `skin_df` DataFrame.

2. **Image Visualization**:
   - We sample **5 images** from each category (`dx`) and display them in a grid.
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

These visualizations help to better understand the dataset’s structure and guide further analysis.



## Future Steps:
- **Data Augmentation**: To enhance model performance, we can implement augmentation techniques such as rotation, flipping, and zooming.
- **Model Training**: We plan to apply machine learning or deep learning models for skin lesion classification based on this dataset.
- **Evaluation and Metrics**: We will evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.
