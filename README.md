# 🚀 Bitcoin Price Prediction Dashboard

An interactive web application for predicting Bitcoin prices using machine learning algorithms, specifically K-Nearest Neighbors (KNN). The dashboard provides real-time price analysis, technical indicators, and future price predictions.

## ✨ Features

- **Real-time Price Analysis**: View current Bitcoin prices and historical trends
- **Price Predictions**: Get future price predictions using KNN algorithm
- **Technical Indicators**: 
  - Moving Averages (7-day and 30-day)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
- **Interactive Charts**: Dark-themed, interactive candlestick charts with hover details
- **Model Performance**: Track and evaluate prediction accuracy
- **Customizable Settings**: Adjust prediction horizons and chart periods

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BTC-using-KNN.git
cd BTC-using-KNN
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the sidebar to:
   - Select prediction horizon
   - Choose models
   - Adjust chart display period

## 📊 Technical Stack

- **Frontend**: Streamlit
- **Data Visualization**: Plotly
- **Machine Learning**: 
  - scikit-learn (KNN)
  - pandas (Data Processing)
  - numpy (Numerical Operations)
- **Technical Analysis**: ta-lib

## 📁 Project Structure

```
BTC-using-KNN/
├── app.py                 # Main Streamlit application
├── bitcoin_prediction.py  # Price prediction logic
├── config.yaml           # Configuration settings
├── requirements.txt      # Project dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

## 🔧 Configuration

Adjust prediction parameters in `config.yaml`:
- Prediction horizons
- Technical indicators
- Model parameters
- Data fetching settings

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data provided by cryptocurrency APIs
- Built with Streamlit's amazing framework
- Inspired by the crypto trading community KNN regression for price prediction
- Includes performance metrics and visualization

## Requirements
- Python 3.7+
- Required packages listed in requirements.txt
## Setup
1. Create and activate virtual environment:
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the prediction script:
   ```bash
   python bitcoin_prediction.py
   ```

4. Deactivate virtual environment when done:
   ```bash
   deactivate
   ```

## Output
- The script will display model performance metrics
- A visualization of actual vs predicted prices will be saved as 'prediction_results.png'
