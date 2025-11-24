# Cardiovascular Disease Risk Predictor

A machine learning web application for predicting the risk of coronary artery (CAD) and cardiovascular disease. Built with Streamlit and a Random Forest model trained on a large-scale dataset.

## Features

- Interactive web interface for patient health data input
- Live prediction of CAD risk probability
- Modern, responsive design with clear visuals
- Suitable for educational, research, or clinical prototype use

## Demo

![App Screenshot](https://cdn-icons-png.flaticon.com/512/306 Started

### Prerequisites

- Python 3.7+
- The dependencies listed in `requirements.txt` (Streamlit, pandas, scikit-learn, numpy)

### Installation

1. Clone this repository:
    ```
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```
2. (Optional) Create and activate a virtual environment:
    ```
    python -m venv venv
    source venv/bin/activate        # On Windows: venv\Scripts\activate
    ```
3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

### Running the App Locally

```bash
streamlit run app.py
```

### Deploying Publicly

For public access, deploy on [Streamlit Cloud](https://streamlit.io/cloud) or host on a cloud VM.  
See “Deployment” section for details.

## Files

| File                              | Purpose                                      |
|------------------------------------|----------------------------------------------|
| app.py                            | Streamlit web app source                     |
| rf_cad_model.pkl                   | Trained Random Forest model for prediction   |
| cardiovascular_disease_cleaned.csv | Cleaned dataset used for training/testing    |
| requirements.txt                   | Python dependencies                         |

## Model

- Trained on Cardiovascular Disease Dataset from Kaggle
- Features: age, gender, height, weight, blood pressure, cholesterol, glucose, smoking/alcohol/physical activity indicators

## Deployment

- Push code and files to a public GitHub repo
- Use Streamlit Cloud for one-click deployment and public link sharing

## License

This project is open-source for educational and research use. See `LICENSE` for details.

## Acknowledgments

- Cardiovascular Disease dataset: [Kaggle link](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- [Streamlit](https://streamlit.io)
- [Flaticon medical icons](https://www.flaticon.com/)

