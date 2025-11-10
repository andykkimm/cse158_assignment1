import gzip
import numpy as np
from collections import defaultdict
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import string

def readGz(path):
    """Read gzipped JSON file."""
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    """Read gzipped CSV file."""
    f = gzip.open(path, 'rt')
    f.readline()  # Skip header
    for l in f:
        yield l.strip().split(',')

print("Loading training data...")

# ============================================================================
# TASK 1: RATING PREDICTION
# ============================================================================
print("\n=== Task 1: Rating Prediction ===")

# Load interaction data
allRatings = []
userRatings = defaultdict(list)
itemRatings = defaultdict(list)
interactions = []

for user, book, r in readCSV("train_Interactions.csv.gz"):
    r = int(r)
    allRatings.append(r)
    userRatings[user].append(r)
    itemRatings[book].append(r)
    interactions.append((user, book, r))

# Compute global statistics
globalAverage = sum(allRatings) / len(allRatings)
print(f"Global average rating: {globalAverage:.3f}")

# Compute user and item averages
userAverage = {}
for u in userRatings:
    userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

itemAverage = {}
for i in itemRatings:
    itemAverage[i] = sum(itemRatings[i]) / len(itemRatings[i])

# Compute user and item biases using simpler approach
print("Computing biases...")
alpha = globalAverage
userBias = defaultdict(float)
itemBias = defaultdict(float)

# Compute item biases first
for item in itemRatings:
    itemBias[item] = (sum(itemRatings[item]) / len(itemRatings[item])) - alpha

# Compute user biases
for user in userRatings:
    userBias[user] = (sum(userRatings[user]) / len(userRatings[user])) - alpha

print("Generating rating predictions...")
predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u, b = l.strip().split(',')

    # Prediction with biases
    prediction = alpha + userBias[u] + itemBias[b]

    # Clip to valid range [1, 5]
    prediction = max(1, min(5, prediction))

    predictions.write(u + ',' + b + ',' + str(prediction) + '\n')

predictions.close()
print("Rating predictions saved to predictions_Rating.csv")

# ============================================================================
# TASK 2: READ PREDICTION
# ============================================================================
print("\n=== Task 2: Read Prediction ===")

# Build user-item interaction matrix
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)

for user, book, _ in readCSV("train_Interactions.csv.gz"):
    usersPerItem[book].add(user)
    itemsPerUser[user].add(book)

# Compute book popularity
bookCount = defaultdict(int)
totalRead = 0

for user, book, _ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

# Sort books by popularity
mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort(reverse=True)

# Get popular books that account for ~50% of reads
popularBooks = set()
count = 0
for ic, book in mostPopular:
    count += ic
    popularBooks.add(book)
    if count > totalRead / 2:
        break

# Jaccard similarity function
def jaccard(s1, s2):
    """Compute Jaccard similarity between two sets."""
    if len(s1) == 0 or len(s2) == 0:
        return 0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union if union > 0 else 0

print("Generating read predictions...")
predictions = open("predictions_Read.csv", 'w')
threshold = 0.5  # Decision threshold

for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u, b = l.strip().split(',')

    # Features for prediction
    score = 0.0

    # Feature 1: Book popularity
    if b in popularBooks:
        score += 0.5

    # Feature 2: User has read similar books (Jaccard similarity)
    if u in itemsPerUser and b in usersPerItem:
        user_books = itemsPerUser[u]
        book_users = usersPerItem[b]

        # Check similarity with other users who read this book
        similarities = []
        for other_user in book_users:
            if other_user != u and other_user in itemsPerUser:
                sim = jaccard(user_books, itemsPerUser[other_user])
                similarities.append(sim)

        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            score += avg_sim * 0.5

    # Feature 3: User activity level
    if u in itemsPerUser:
        user_activity = len(itemsPerUser[u]) / 100  # Normalize
        score += min(user_activity, 0.2)

    # Make prediction
    prediction = 1 if score > threshold else 0
    predictions.write(u + ',' + b + ',' + str(prediction) + '\n')

predictions.close()
print("Read predictions saved to predictions_Read.csv")

# ============================================================================
# TASK 3: CATEGORY PREDICTION
# ============================================================================
print("\n=== Task 3: Category Prediction ===")

# Category mapping
catDict = {
    "children": 0,
    "comics_graphic": 1,
    "fantasy_paranormal": 2,
    "mystery_thriller_crime": 3,
    "young_adult": 4
}

# Reverse mapping
catNames = {v: k for k, v in catDict.items()}

# Load training data for category prediction
print("Loading category training data...")
train_texts = []
train_labels = []
train_ratings = []

for review in readGz("train_Category.json.gz"):
    review_text = review['review_text']
    genre_id = review['genreID']
    rating = review['rating']

    train_texts.append(review_text)
    train_labels.append(genre_id)
    train_ratings.append(rating)

print(f"Loaded {len(train_texts)} training reviews")

# Text preprocessing
def preprocess_text(text):
    """Basic text preprocessing."""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Preprocess all texts
train_texts_processed = [preprocess_text(text) for text in train_texts]

# Create TF-IDF features
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts_processed)

print(f"Feature matrix shape: {X_train.shape}")

# Train classifier
print("Training classifier...")
classifier = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial', solver='lbfgs', random_state=42)
classifier.fit(X_train, train_labels)

train_acc = classifier.score(X_train, train_labels)
print(f"Training accuracy: {train_acc:.3f}")

# Generate predictions
print("Generating category predictions...")
predictions = open("predictions_Category.csv", 'w')
predictions.write("userID,reviewID,prediction\n")

for review in readGz("test_Category.json.gz"):
    user_id = review['user_id']
    review_id = review['review_id']
    review_text = review['review_text']

    # Preprocess and vectorize
    text_processed = preprocess_text(review_text)
    X_test = vectorizer.transform([text_processed])

    # Predict
    pred = classifier.predict(X_test)[0]

    predictions.write(user_id + ',' + review_id + ',' + str(pred) + '\n')

predictions.close()
print("Category predictions saved to predictions_Category.csv")

print("\n=== All predictions completed! ===")
print("Generated files:")
print("  - predictions_Rating.csv")
print("  - predictions_Read.csv")
print("  - predictions_Category.csv")
