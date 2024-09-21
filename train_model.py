import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
dataset = pd.read_csv("datasetfinal.csv")

# Use only the relevant columns
features = ["movie_title", "director_name", "actor_1_name"]
target = "Hit/Flop"

# Encoding categorical data
nameencoder = LabelEncoder()
actor1encoder = LabelEncoder()
movieencoder = LabelEncoder()  # Add encoder for movie titles

# Transforming the relevant columns
dataset['director_name'] = nameencoder.fit_transform(dataset['director_name'])
dataset['actor_1_name'] = actor1encoder.fit_transform(dataset['actor_1_name'])
dataset['movie_title'] = movieencoder.fit_transform(dataset['movie_title'])  # Encode movie titles

# Prepare X and y
X = dataset[features]
y = dataset[target]

# Encoding the target variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Save model and transformers
joblib.dump(model, 'Model/model.pkl')
joblib.dump(scaler, 'Model/scaler.pkl')
joblib.dump(nameencoder, 'Model/nameencoder.pkl')
joblib.dump(actor1encoder, 'Model/actor1encoder.pkl')
joblib.dump(movieencoder, 'Model/movieencoder.pkl')  # Save the movie encoder
joblib.dump(labelencoder_y, 'Model/labelencoder_y.pkl')
