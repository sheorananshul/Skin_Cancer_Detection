# Skin Lesion Classification

## Dataset Overview
This project utilizes the **HAM10000** dataset, a collection of skin lesion images. The dataset contains **10,000 images** labeled with the diagnosis (`dx`) for each skin lesion. The images in the dataset come from multiple categories, each representing a different type of skin condition. The images are in `.jpg` format and vary in size, with some being resized for visualization.

## Steps Taken So Far

1. **Dataset Loading and Preprocessing**:
   - The dataset's metadata file (`HAM10000_metadata.csv`) is used to load information about each image.
   - The image paths are mapped from the metadata, allowing us to locate each image in the file system.
   - Each image is read using the `PIL` library and stored in the `skin_df` DataFrame.

2. **Image Visualization**:
   - We sample **5 images** from each category (`dx`) and display them in a grid.
   - Images are displayed at their original resolution without resizing to maintain their quality.
   - **Matplotlib** is used to create subplots and visualize the sampled images, and we save the final plot at **DPI 500** to ensure high-quality output.

3. **Category Distribution**:
   - We display the distribution of categories (`dx`) to understand the balance between different skin lesion types.
   - The data is grouped by the `dx` column, and a specified number of samples per category are visualized.

4. **Saving Visuals**:
   - The final image visualization grid is saved as `category_samples.png`, with high resolution for detailed analysis.

## Future Steps:
- **Data Augmentation**: To enhance model performance, we can implement augmentation techniques such as rotation, flipping, and zooming.
- **Model Training**: We plan to apply machine learning or deep learning models for skin lesion classification based on this dataset.
- **Evaluation and Metrics**: We will evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.
