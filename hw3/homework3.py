from collections import Counter, defaultdict
import gzip
import math
import scipy.optimize
import numpy as np
import string
from sklearn import linear_model
import random

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

def Jaccard(s1, s2):
    intersection_size = len(s1.intersection(s2))
    union_size = len(s1.union(s2))
    if union_size > 0:
        return intersection_size/union_size
    return 0

def getGlobalAverage(trainRatings):
    if isinstance(trainRatings[0], (list, tuple)):
        rating_values = [r[2] for r in trainRatings]
    else:
        rating_values = trainRatings

    return np.mean(rating_values)

def trivialValidMSE(ratingsValid, globalAverage):
    if isinstance(ratingsValid[0], (list, tuple)):
        actual_ratings = np.array([r[2] for r in ratingsValid])
    else:
        actual_ratings = np.array(ratingsValid)

    predicted_ratings = np.full(len(actual_ratings), globalAverage)
    mean_sq_error = np.mean((actual_ratings - predicted_ratings) ** 2)
    return mean_sq_error

def alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb):
    sum_residuals = 0.0
    for (u, i, r) in ratingsTrain:
        sum_residuals += (r - (betaU[u] + betaI[i]))
    newAlpha = sum_residuals / len(ratingsTrain)
    return newAlpha

def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb):
    updated_betaU = {}
    for u, ratings in ratingsPerUser.items():
        sum_residuals = 0.0
        for (i, r) in ratings:
            sum_residuals += (r - (alpha + betaI[i]))
        updated_betaU[u] = sum_residuals / (lamb + len(ratings))
    return updated_betaU

def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb):
    updated_betaI = {}
    for i, ratings in ratingsPerItem.items():
        sum_residuals = 0.0
        for (u, r) in ratings:
            sum_residuals += (r - (alpha + betaU[u]))
        updated_betaI[i] = sum_residuals / (lamb + len(ratings))
    return updated_betaI

def msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb):
    error_sum = 0.0
    for (u, i, r) in ratingsTrain:
        prediction = alpha + betaU[u] + betaI[i]
        error_sum += (r - prediction) ** 2
    
    mean_sq_error = error_sum / len(ratingsTrain)

    regularization = sum(b ** 2 for b in betaU.values()) + sum(b ** 2 for b in betaI.values())
    objective = mean_sq_error + lamb * regularization

    return mean_sq_error, objective

def validMSE(ratingsValid, alpha, betaU, betaI):
    error_sum = 0.0
    for (u, i, r) in ratingsValid:
        prediction = alpha + betaU.get(u, 0) + betaI.get(i, 0)
        error_sum += (r - prediction) ** 2

    mean_sq_error = error_sum / len(ratingsValid)
    return mean_sq_error

def goodModel(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI):
    iterations = 10  

    for iteration in range(iterations):
        alpha = alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb=1)
        betaU = betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb=1)
        betaI = betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb=1)

        mean_sq_error, regularized_obj = msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb=1)
        print(f"Iteration {iteration+1}: MSE={mean_sq_error:.6f}, RegObj={regularized_obj:.6f}")

    return alpha, betaU, betaI

def writePredictionsRating(alpha, betaU, betaI):
    # Write your predictions to a file that you can submit
    output_file = open("predictions_Rating.csv", 'w')
    for line in open("pairs_Rating.csv"):
        if line.startswith("userID"):
            output_file.write(line)
            continue
        u,b = line.strip().split(',')
        user_bias = 0
        item_bias = 0
        if u in betaU:
            user_bias = betaU[u]
        if b in betaI:
            item_bias = betaI[b]
        _ = output_file.write(u + ',' + b + ',' + str(alpha + user_bias + item_bias) + '\n')

    output_file.close()

def generateValidation(allRatings, ratingsValid):
    user_book_map = defaultdict(set)
    book_set = set()
    for (u, b, _) in allRatings:
        user_book_map[u].add(b)
        book_set.add(b)
    
    positive_pairs = [(u, b) for (u, b, _) in ratingsValid]
    
    negative_pairs = []
    for (u, b, _) in ratingsValid:
        unread_books = list(book_set - user_book_map[u])
        if not unread_books:
            continue  
        negative_book = random.choice(unread_books)
        negative_pairs.append((u, negative_book))
    
    pair_count = min(len(positive_pairs), len(negative_pairs))
    positive_pairs = positive_pairs[:pair_count]
    negative_pairs = negative_pairs[:pair_count]

    return positive_pairs, negative_pairs

def baseLineStrategy(mostPopular, totalRead):
    predicted_positive = set()

    # Compute the set of items for which we should return "True"
    # This is the same strategy implemented in the baseline code for Assignment 1
    predicted_positive = set()
    accumulated_count = 0
    for item_count, item in mostPopular:
        accumulated_count += item_count
        predicted_positive.add(item)
        if accumulated_count > totalRead/2: break

    return predicted_positive

def improvedStrategy(mostPopular, totalRead):
    predicted_positive = set()
    accumulated_count = 0
    cutoff = 0.6  

    for item_count, item in mostPopular:
        accumulated_count += item_count
        predicted_positive.add(item)
        if accumulated_count > totalRead * cutoff:
            break

    return predicted_positive

def evaluateStrategy(return1, readValid, notRead):
    num_correct = 0
    num_total = len(readValid) + len(notRead)
    for (u, b) in readValid:
        if b in return1:
            num_correct += 1

    for (u, b) in notRead:
        if b not in return1:
            num_correct += 1

    accuracy = num_correct / num_total
    return accuracy

def jaccardThresh(u,b,ratingsPerItem,ratingsPerUser):
    highest_similarity = 0
    if u in ratingsPerUser:
        for other_book in ratingsPerUser[u]:
            if other_book == b:
                continue
            if b in ratingsPerItem and other_book in ratingsPerItem:
                similarity = Jaccard(ratingsPerItem[b], ratingsPerItem[other_book])
                if similarity > highest_similarity:
                    highest_similarity = similarity

    if highest_similarity > 0.013 or len(ratingsPerItem.get(b, [])) > 40:
        return 1
    return 0

def writePredictionsRead(ratingsPerItem, ratingsPerUser):
    output_file = open("predictions_Read.csv", 'w')
    for line in open("pairs_Read.csv"):
        if line.startswith("userID"):
            output_file.write(line)
            continue
        u,b = line.strip().split(',')
        prediction = jaccardThresh(u,b,ratingsPerItem,ratingsPerUser)
        _ = output_file.write(u + ',' + b + ',' + str(prediction) + '\n')

    output_file.close()

def featureCat(datum, words, wordId, wordSet):
    review_text = datum['review_text'].lower().translate(str.maketrans('', '', string.punctuation))
    feature_vector = [0] * len(words)
    word_tokens = review_text.split()
    for token in word_tokens:
        if token in wordSet:
            index = wordId[token]
            feature_vector[index] += 1
    feature_vector.append(1)
    return feature_vector

def betterFeatures(data):
    vocabulary_counts = Counter()
    for d in data:
        review_text = d['review_text'].lower().translate(str.maketrans('', '', string.punctuation))
        word_tokens = review_text.split()
        vocabulary_counts.update(word_tokens)
    
    frequent_words = [w for w, _ in vocabulary_counts.most_common(1000)]
    word_to_id = {w: i for i, w in enumerate(frequent_words)}
    word_vocabulary = set(frequent_words)
    
    feature_matrix = []
    for d in data:
        review_text = d['review_text'].lower().translate(str.maketrans('', '', string.punctuation))
        word_tokens = review_text.split()
        feature_vector = [0] * len(frequent_words)
        for token in word_tokens:
            if token in word_vocabulary:
                feature_vector[word_to_id[token]] += 1
        feature_vector.append(1)  
        feature_matrix.append(feature_vector)

    return feature_matrix

def runOnTest(data_test, mod):
    test_features = [featureCat(d) for d in data_test]
    test_predictions = mod.predict(test_features)

def writePredictionsCategory(pred_test):
    output_file = open("predictions_Category.csv", 'w')
    position = 0

    for line in open("/Users/victorchen/Desktop/cse158/hw3/assignment1/pairs_Category.csv"):
        if line.startswith("userID"):
            output_file.write(line)
            continue
        u,b = line.strip().split(',')
        _ = output_file.write(u + ',' + b + ',' + str(pred_test[position]) + '\n')
        position += 1

    output_file.close()
