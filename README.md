# ML_classification: Music Genre Prediction

## Description
This project predicts music genres (Rock, Hip-Hop, Pop) based on audio features like Energy, Danceability, and Loudness using Logistic Regression from Scikit-learn. It’s a fun, energetic, and hardcore example of machine learning classification, perfect for music lovers and data enthusiasts!

## Data
- **Source**: Synthetic dataset generated for demonstration (150 tracks: 50 Rock, 50 Hip-Hop, 50 Pop)
- **Features**: 
  - Energy (0–1, how energetic the track is)
  - Danceability (0–1, how danceable the track is)
  - Loudness (dB, how loud the track is)
- **Target**: Genre (Rock, Hip-Hop, Pop, encoded as 0, 1, 2)

## Results
- Accuracy on test data: ~80–90%
- Confusion Matrix shows how well the model distinguishes between genres

## Visualizations
- **Confusion Matrix**: Shows prediction accuracy for each genre (Rock, Hip-Hop, Pop)
- **Energy vs Danceability Plot**: Visualizes how genres cluster based on energy and danceability

## Requirements
- Python 3.x
- Libraries: numpy, pandas, matplotlib, scikit-learn

## How to Run
1. Install dependencies: `pip install numpy pandas matplotlib scikit-learn`
2. Run the script: `python music_genre_prediction.py`

## Visualizations
- [confusion_matrix.png](https://github.com/AndreySukhanov/ML_classification/blob/16a241632f3bcd1660d05f6e8c582fe80485011c/confusion_matrix.png)
- [energy_danceability.png](https://github.com/AndreySukhanov/ML_classification/blob/4a4356bcc9be05ca9c8f58dba0e589f3c4f7c5ea/energy_danceability.png)

## Notes
This is a synthetic dataset for educational purposes. For real-world applications, you can use the Spotify Tracks Dataset from Kaggle or the Spotify API.
