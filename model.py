import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import time
import psutil

# Record start time
start_time = time.time()

# Download NLTK resources (only required once)
nltk.download('stopwords')
nltk.download('punkt')

# Load your dataset with explicit encoding
dataset_path = 'Path to your dataset'
df = pd.read_csv(dataset_path)

def preprocess_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    
    # Convert words to lowercase
    tokens = [word.lower() for word in tokens]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Reconstruct the text from tokens
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Apply the preprocessing function to your DataFrame
df['clean_text'] = df['review'].apply(preprocess_text)

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis on each review and calculate accuracy
predictions = []
actual_labels = []
for i, row in df.iterrows():
    review = row['clean_text']
    label = row['label']
    
    sentiment_score = sia.polarity_scores(review)
    if sentiment_score['compound'] >= 0.05:
        predicted_sentiment = 'pos'
    elif sentiment_score['compound'] <= -0.05:
        predicted_sentiment = 'neg'
    else:
        predicted_sentiment = 'neutral'
    
    predictions.append(predicted_sentiment)
    actual_labels.append(label)

# Creating binary labels for the confusion matrix
binary_actual_labels = ['pos' if label == 'pos' else 'neg' for label in actual_labels]
binary_predictions = ['pos' if pred == 'pos' else 'neg' for pred in predictions]

# Calculate confusion matrix
conf_matrix = confusion_matrix(binary_actual_labels, binary_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate precision
precision = precision_score(binary_actual_labels, binary_predictions, average='binary', pos_label='pos')
print("Precision:", precision)

# Calculate recall
recall = recall_score(binary_actual_labels, binary_predictions, average='binary', pos_label='pos')
print("Recall:", recall)

# Record end time
end_time = time.time()

# Calculate accuracy
accuracy = accuracy_score(binary_actual_labels, binary_predictions)
print("Accuracy:", accuracy)

# Calculate F1 score
f1 = f1_score(binary_actual_labels, binary_predictions, average='binary', pos_label='pos')
print("F1 Score:", f1)

# Calculate running time
running_time = end_time - start_time
print("Total running time:", running_time, "seconds.")

# Add predicted sentiment column to the DataFrame
df['predicted_sentiment'] = predictions

# Save results to CSV file
output_path = 'Output directory path [Eg. "Path/output.csv".]'
df.to_csv(output_path, index=False)
print("Results saved at:", output_path)

# Function to print CPU and memory usage
def print_system_resources():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    
    print("CPU Usage:", cpu_percent, "%")
    print("Memory Usage:")
    print("  Total:", round(memory_info.total / (1024 * 1024 * 1024), 2), "GB")
    print("  Available:", round(memory_info.available / (1024 * 1024 * 1024), 2), "GB")
    print("  Used:", round(memory_info.used / (1024 * 1024 * 1024), 2), "GB")
    print("  Free:", round(memory_info.free / (1024 * 1024 * 1024), 2), "GB")

# Print system resources
print_system_resources()
