"""
CSE 158 Assignment 1: Goodreads Recommender Systems
Author: Assignment Solution
Date: November 2025

This file contains implementations for three prediction tasks:
1. Read Prediction: Predict whether a user will read a book
2. Category Prediction: Predict the genre of a book from review text
3. Rating Prediction: Predict star ratings for user-book pairs
"""

import gzip
import numpy as np
from collections import defaultdict
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings('ignore')

# Utility functions to read data
def readGz(path):
    """Read gzipped JSON file line by line"""
    for l in gzip.open(path, 'rt', encoding='utf-8'):
        yield eval(l)

def readCSV(path):
    """Read gzipped CSV file"""
    f = gzip.open(path, 'rt')
    f.readline()  # Skip header
    for l in f:
        yield l.strip().split(',')

###############################################################################
# TASK 1: READ PREDICTION
###############################################################################
print("=" * 80)
print("TASK 1: READ PREDICTION")
print("=" * 80)

# Load training data
print("Loading training data...")
userBooks = defaultdict(set)
bookUsers = defaultdict(set)
allBooks = set()
allUsers = set()
bookCount = defaultdict(int)
userCount = defaultdict(int)

for user, book, rating in readCSV("train_Interactions.csv.gz"):
    userBooks[user].add(book)
    bookUsers[book].add(user)
    allBooks.add(book)
    allUsers.add(user)
    bookCount[book] += 1
    userCount[user] += 1

print(f"Loaded {len(allUsers)} users and {len(allBooks)} books")

# Calculate book popularity
totalInteractions = sum(bookCount.values())
bookPopularity = {book: count / totalInteractions for book, count in bookCount.items()}

# Find most popular books (for baseline comparison)
mostPopular = sorted(bookCount.items(), key=lambda x: x[1], reverse=True)
popularBooks = set()
count = 0
for book, c in mostPopular:
    count += c
    popularBooks.add(book)
    if count > totalInteractions / 2:
        break

# Jaccard similarity function
def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    if len(set1) == 0 or len(set2) == 0:
        return 0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# Load test pairs
print("Making read predictions...")
predictions_read = open("predictions_Read.csv", 'w')
predictions_read.write("userID,bookID,prediction\n")

# Read prediction model
threshold = 0.5
read_count = 0
total_count = 0

for line in open("pairs_Read.csv"):
    if line.startswith("userID"):
        continue

    user, book = line.strip().split(',')
    total_count += 1

    # Feature 1: Book popularity
    pop_score = bookPopularity.get(book, 0)

    # Feature 2: User activity level
    user_activity = len(userBooks.get(user, set()))

    # Feature 3: Jaccard similarity with user's read books
    if user in userBooks:
        # Calculate similarity between this book's readers and user's books
        book_readers = bookUsers.get(book, set())
        user_books = userBooks[user]

        # Find similar users who read this book
        similar_score = 0
        for user_book in list(user_books)[:50]:  # Limit for efficiency
            book_readers_other = bookUsers.get(user_book, set())
            sim = jaccard_similarity(book_readers, book_readers_other)
            similar_score += sim

        if len(user_books) > 0:
            similar_score /= min(len(user_books), 50)
    else:
        similar_score = 0

    # Feature 4: Check if book is in popular set
    is_popular = 1 if book in popularBooks else 0

    # Combined score
    score = (0.4 * is_popular +
             0.3 * pop_score * 100 +
             0.2 * similar_score +
             0.1 * min(user_activity / 50, 1))

    # Predict
    if score > threshold:
        prediction = 1
        read_count += 1
    else:
        prediction = 0

    predictions_read.write(f"{user},{book},{prediction}\n")

predictions_read.close()

# Adjust to ensure 50/50 split
print(f"Initial predictions: {read_count}/{total_count} = {read_count/total_count:.2%} read")

# Re-run with threshold adjustment to get closer to 50%
print("Adjusting threshold for 50/50 split...")
scores = []

for line in open("pairs_Read.csv"):
    if line.startswith("userID"):
        continue

    user, book = line.strip().split(',')

    pop_score = bookPopularity.get(book, 0)
    user_activity = len(userBooks.get(user, set()))

    if user in userBooks:
        book_readers = bookUsers.get(book, set())
        user_books = userBooks[user]
        similar_score = 0
        for user_book in list(user_books)[:50]:
            book_readers_other = bookUsers.get(user_book, set())
            sim = jaccard_similarity(book_readers, book_readers_other)
            similar_score += sim
        if len(user_books) > 0:
            similar_score /= min(len(user_books), 50)
    else:
        similar_score = 0

    is_popular = 1 if book in popularBooks else 0
    score = (0.4 * is_popular + 0.3 * pop_score * 100 +
             0.2 * similar_score + 0.1 * min(user_activity / 50, 1))
    scores.append((user, book, score))

# Sort by score and take top 50%
scores.sort(key=lambda x: x[2], reverse=True)
threshold_idx = len(scores) // 2
threshold_score = scores[threshold_idx][2]

# Write final predictions
predictions_read = open("predictions_Read.csv", 'w')
predictions_read.write("userID,bookID,prediction\n")

for user, book, score in scores:
    prediction = 1 if score >= threshold_score else 0
    predictions_read.write(f"{user},{book},{prediction}\n")

predictions_read.close()
print("Read predictions saved to predictions_Read.csv")

###############################################################################
# TASK 2: CATEGORY PREDICTION
###############################################################################
print("\n" + "=" * 80)
print("TASK 2: CATEGORY PREDICTION")
print("=" * 80)

# Category mapping
catDict = {
    "children": 0,
    "comics_graphic": 1,
    "fantasy_paranormal": 2,
    "mystery_thriller_crime": 3,
    "young_adult": 4
}

# Reverse mapping
id2cat = {v: k for k, v in catDict.items()}

# Load training data
print("Loading category training data...")
train_texts = []
train_labels = []
train_ratings = []

for review in readGz("train_Category.json.gz"):
    text = review.get('review_text', '')
    genre_id = review.get('genreID', 2)
    rating = review.get('rating', 3)

    train_texts.append(text)
    train_labels.append(genre_id)
    train_ratings.append(rating)

print(f"Loaded {len(train_texts)} training reviews")

# Text preprocessing and feature extraction
print("Training TF-IDF vectorizer and classifier...")

# TF-IDF with character and word n-grams
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    strip_accents='unicode',
    lowercase=True,
    stop_words='english'
)

X_train = vectorizer.fit_transform(train_texts)

# Train classifier (Logistic Regression works well for text classification)
classifier = LogisticRegression(
    max_iter=1000,
    C=1.0,
    multi_class='multinomial',
    solver='lbfgs',
    random_state=42
)
classifier.fit(X_train, train_labels)

print(f"Training accuracy: {classifier.score(X_train, train_labels):.4f}")

# Make predictions on test data
print("Making category predictions...")
predictions_category = open("predictions_Category.csv", 'w')
predictions_category.write("userID,reviewID,prediction\n")

test_count = 0
for review in readGz("test_Category.json.gz"):
    user_id = review['user_id']
    review_id = review['review_id']
    text = review.get('review_text', '')

    # Transform text and predict
    X_test = vectorizer.transform([text])
    prediction = classifier.predict(X_test)[0]

    predictions_category.write(f"{user_id},{review_id},{prediction}\n")
    test_count += 1

predictions_category.close()
print(f"Category predictions saved to predictions_Category.csv ({test_count} predictions)")

###############################################################################
# TASK 3: RATING PREDICTION
###############################################################################
print("\n" + "=" * 80)
print("TASK 3: RATING PREDICTION")
print("=" * 80)

# Load training data for rating prediction
print("Loading interaction data for rating prediction...")
allRatings = []
userRatings = defaultdict(list)
bookRatings = defaultdict(list)
interactions = []

for user, book, rating in readCSV("train_Interactions.csv.gz"):
    r = int(rating)
    allRatings.append(r)
    userRatings[user].append(r)
    bookRatings[book].append(r)
    interactions.append((user, book, r))

# Calculate global average
globalAverage = sum(allRatings) / len(allRatings)
print(f"Global average rating: {globalAverage:.3f}")

# Calculate user and book biases
userBias = {}
for user in userRatings:
    userBias[user] = sum(userRatings[user]) / len(userRatings[user]) - globalAverage

bookBias = {}
for book in bookRatings:
    bookBias[book] = sum(bookRatings[book]) / len(bookRatings[book]) - globalAverage

# Calculate damped/regularized averages to avoid overfitting
def dampedAverage(values, globalAvg, damping=5):
    """Calculate damped average to handle users/items with few ratings"""
    if len(values) == 0:
        return globalAvg
    return (sum(values) + damping * globalAvg) / (len(values) + damping)

userAverage = {}
for user in userRatings:
    userAverage[user] = dampedAverage(userRatings[user], globalAverage, damping=10)

bookAverage = {}
for book in bookRatings:
    bookAverage[book] = dampedAverage(bookRatings[book], globalAverage, damping=10)

# Make rating predictions
print("Making rating predictions...")
predictions_rating = open("predictions_Rating.csv", 'w')
predictions_rating.write("userID,bookID,prediction\n")

for line in open("pairs_Rating.csv"):
    if line.startswith("userID"):
        continue

    user, book = line.strip().split(',')

    # Bias-based prediction model
    prediction = globalAverage

    # Add user bias if we've seen this user
    if user in userBias:
        prediction += 0.5 * userBias[user]

    # Add book bias if we've seen this book
    if book in bookBias:
        prediction += 0.5 * bookBias[book]

    # Alternative: use damped averages
    if user in userAverage and book in bookAverage:
        # Weighted combination of user avg, book avg, and bias model
        user_pred = userAverage[user]
        book_pred = bookAverage[book]
        bias_pred = prediction

        # Weight based on amount of data
        user_weight = min(len(userRatings[user]) / 20, 0.5)
        book_weight = min(len(bookRatings[book]) / 20, 0.5)

        prediction = (user_weight * user_pred +
                     book_weight * book_pred +
                     (1 - user_weight - book_weight) * bias_pred)
    elif user in userAverage:
        prediction = 0.7 * userAverage[user] + 0.3 * globalAverage
    elif book in bookAverage:
        prediction = 0.7 * bookAverage[book] + 0.3 * globalAverage

    # Clip to valid range [1, 5]
    prediction = max(1, min(5, prediction))

    predictions_rating.write(f"{user},{book},{prediction}\n")

predictions_rating.close()
print("Rating predictions saved to predictions_Rating.csv")

print("\n" + "=" * 80)
print("ALL TASKS COMPLETED!")
print("=" * 80)
print("\nGenerated files:")
print("  - predictions_Read.csv")
print("  - predictions_Category.csv")
print("  - predictions_Rating.csv")
print("\nYou can now upload these files to Gradescope.")
