
# Multi-Class Eye Disease Classification using CNN

This project presents a Convolutional Neural Network (CNN) model developed to classify images into 12 different classes of eye diseases. The project includes a user-friendly interface built with Streamlit, allowing users to upload eye images and predict the disease.

## 🌐 [**Live Demo on Streamlit**](https://eye-disease-classification-cnn-m59v9noqdytez6s8obcszw.streamlit.app/)

-You can interact with the model directly through the Streamlit app by clicking the link above.:

-You can use [sample images](https://github.com/mujeebrahman022/Eye-Disease-Classification-CNN/tree/main/Sample_images) for prediction while using the streamlit.:

## Project Structure

The project is organized into the following key files:

- **app.py**: The main script to run the Streamlit application, providing a simple UI for users to interact with the model.
- **eye_model.keras**: The trained CNN model used for classification.
- **requirements.txt**: A list of dependencies required to run the project, ensuring that the necessary Python packages are installed.
- **Capstone-EyeProject.ipynb**: A Jupyter Notebook containing the code for data preprocessing, model training, evaluation, and saving the trained model.
- **Dataset**: [**Dataset Link**](#) contains the images used for training and validating the CNN model.

## Getting Started

### Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.7 or higher
- Git

### Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/multi-eye-classification.git
   cd multi-eye-classification
   ```

2. **Set up a virtual environment and install dependencies:**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scriptsctivate`
   pip install -r requirements.txt
   ```

### Running the Application

Once the setup is complete, you can run the Streamlit application:

```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501` in your web browser.

## Using the Application

- **Upload an Image**: Drag and drop or browse to upload an eye image (supported formats: JPG, JPEG, PNG).
- **Get Predictions**: The model will analyze the uploaded image and classify it into one of the 12 eye disease categories.
- **View Results**: The predicted disease will be displayed on the interface.

### Mixed Sample Images

A file containing mixed sample images of multiple eye diseases is provided for testing the prediction capabilities of the model. You can use these images directly within the Streamlit app to evaluate how well the model classifies different eye conditions.

## Model Details

The CNN model is designed to handle multi-class classification for 12 different eye diseases. It was trained on a dataset containing images of various eye conditions, and the model file is saved as eye_model.keras.

## Model Training

The model was trained using Python and Keras, with extensive data preprocessing and augmentation techniques applied to improve performance. The entire training process, including model architecture, hyperparameter tuning, and evaluation metrics, is documented in Capstone-EyeProject.ipynb.

## Acknowledgments

Thanks to the open-source community for providing the tools and resources that made this project possible. The dataset utilized in this project was sourced from Kaggle.
