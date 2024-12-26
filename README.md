# README for Modbus Protocol IDS with xNIDS Integration

## Project Overview
This project involves the development of an Intrusion Detection System (IDS) that leverages machine learning to classify Modbus protocol network traffic as safe or malicious. The project also integrates this IDS with the xNIDS explainer model to provide explanations for the classifications made by the IDS.

---

## Project Structure

The project is divided into two main parts:

1. **IDS Model Training and Preprocessing**
2. **Integration with xNIDS Explainer Model**

---

## Files and Directories

- `IDS_training_and_preprocessing.ipynb`: Jupyter notebook for data preprocessing, model training, evaluation, and saving the trained model.
- `xNIDS_integration.ipynb`: Jupyter notebook for integrating the trained IDS model with the xNIDS explainer to provide explanations for malicious classifications.
- `modbus_data_sample.csv`: Sample data used for testing and predictions.
- `modbus_ids_model.pkl`: Serialized IDS model saved using joblib.
- `/xNIDS`: Directory containing the xNIDS explainer code.

---

## Installation and Setup

### Prerequisites
Ensure you have the following software and packages installed:

- Python 3.x
- Jupyter Notebook
- Libraries: `pandas`, `matplotlib`, `seaborn`, `joblib`, `scikit-learn`, `pyshark`, `asgl`, `tensorflow`, `numpy`
- TShark (Wireshark CLI)

### Installation Steps

1. **Install TShark**
   ```bash
   sudo apt-get install -y tshark
   ```

2. **Install Python packages**
   ```bash
   pip install pyshark joblib scikit-learn imbalanced-learn matplotlib seaborn tensorflow asgl
   ```

3. **Clone the xNIDS repository**
   ```bash
   git clone https://github.com/CactiLab/code-xNIDS.git
   mv code-xNIDS xNIDS
   ```

---

## Usage Guide

### Running the IDS Model Training

1. Open `IDS_training_and_preprocessing.ipynb` in Jupyter Notebook.
2. For running the first notebook, upload the following files to Colab:
   - `CnC_uploading_exe_modbus_6RTU_with_operate_labeled.csv`
   - `CnC_uploading_exe_modbus_6RTU_with_operate_labeled.pcap`
3. Follow the cells to:
   - Load and preprocess the dataset.
   - Train the Random Forest classifier.
   - Evaluate model performance with accuracy, confusion matrix, and ROC curve.
   - Save the trained model as `modbus_ids_model.pkl`.

### Integrating with xNIDS

1. Open `xNIDS_integration.ipynb`.
2. Load the `modbus_ids_model.pkl` and `modbus_data_sample.csv` into the `/xNIDS` directory.
3. Use the xNIDS explainer class to analyze the IDS model’s predictions.

---

## Key Features

- **Traffic Analysis**: Visualize packet volume, source, and destination IP distributions.
- **Model Training**: Train a machine learning model with Random Forest and optimize with GridSearchCV.
- **Model Evaluation**: Evaluate using metrics like accuracy, F1-score, confusion matrix, and AUC-ROC.
- **Integration**: Connect IDS predictions with the xNIDS explainer to validate results and check feature importance.

---

## Results and Findings

- The trained IDS model achieved an average accuracy with detailed performance metrics shown in the notebooks.
- The xNIDS integration provides feature-based explanations for why certain traffic was flagged as malicious.
