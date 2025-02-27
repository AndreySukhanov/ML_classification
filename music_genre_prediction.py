import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Generate a synthetic dataset for music genres (Rock, Hip-Hop, Pop)
# Features: Energy, Danceability, Loudness
# Target: Genre (Rock=0, Hip-Hop=1, Pop=2)
np.random.seed(42)  # For reproducibility

n_samples = 150  # 50 tracks per genre
energy = np.concatenate([np.random.normal(0.8, 0.1, 50),  # Rock — high energy
                         np.random.normal(0.7, 0.1, 50),  # Hip-Hop — energetic
                         np.random.normal(0.6, 0.1, 50)])  # Pop — moderate energy
danceability = np.concatenate([np.random.normal(0.4, 0.1, 50),  # Rock — less danceable
                              np.random.normal(0.8, 0.1, 50),  # Hip-Hop — highly danceable
                              np.random.normal(0.7, 0.1, 50)])  # Pop — danceable
loudness = np.concatenate([np.random.normal(-5, 2, 50),  # Rock — loud
                          np.random.normal(-3, 2, 50),  # Hip-Hop — louder
                          np.random.normal(-7, 2, 50)])  # Pop — softer
genres = np.array(['Rock'] * 50 + ['Hip-Hop'] * 50 + ['Pop'] * 50)

# Create DataFrame
df = pd.DataFrame({
    'Energy': energy,
    'Danceability': danceability,
    'Loudness': loudness,
    'Genre': genres
})

# Prepare data
X = df[['Energy', 'Danceability', 'Loudness']].values  # Features
y = pd.Categorical(df['Genre']).codes  # Convert genres to numeric: 0=Rock, 1=Hip-Hop, 2=Pop

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Logistic Regression for multiclass)
model = LogisticRegression(multi_class='ovr', max_iter=1000)  # One-vs-Rest for multiple classes
model.fit(X_train, y_train)
print("\n=== Music Genre Classification Results ===")
print("Accuracy on training data:", model.score(X_train, y_train))
print("Accuracy on test data:", model.score(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)
print("First 5 predictions (0=Rock, 1=Hip-Hop, 2=Pop):", y_pred[:5])
print("Actual values:", y_test[:5])

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Reds)
plt.title("Confusion Matrix for Music Genres")
plt.colorbar()
plt.xticks([0, 1, 2], ['Rock', 'Hip-Hop', 'Pop'])
plt.yticks([0, 1, 2], ['Rock', 'Hip-Hop', 'Pop'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(3):
    for j in range(3):
        plt.text(j, i, conf_matrix[i, j], ha="center", va="center", color="white")
plt.savefig("confusion_matrix.png")  # Save for GitHub
plt.show()

# Additional visualization: Energy vs Danceability by genre
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Energy'], df['Danceability'], c=y, cmap='viridis', alpha=0.6)
plt.xlabel("Energy")
plt.ylabel("Danceability")
plt.title("Music Genres: Energy vs Danceability")
plt.colorbar(scatter, label='Genre (0=Rock, 1=Hip-Hop, 2=Pop)')
plt.grid(True)
plt.savefig("energy_danceability.png")  # Save for GitHub
plt.show()
