from sklearn.feature_selection import SelectKBest, chi2
import jsonlines
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import nltk
from nltk.corpus import stopwords
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pickle

nltk.download('punkt')

# Load stop words
stop_words = set(stopwords.words('english'))

# Define function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into a string
    processed_text = " ".join(tokens)
    return processed_text

def read_and_preprocess_reviews():
    texts = []
    stars = []
    cool = []
    useful = []
    funny = []
    
    counter = 0
    with jsonlines.open('C:/Users/navij/Downloads/yelp_academic_dataset_review.json') as reader:
        for obj in reader:
            text = obj['text']  # Use 'text' key for review text
            star = obj['stars']  # Use 'stars' key for rating
            use = obj['useful'] # Use 'useful' key for useful rating
            fun = obj['funny']  # Use 'funny' key for funny rating
            coo = obj['cool']   # Use 'cool' key for cool rating
            # Preprocess the text
            processed_text = preprocess_text(text)
            texts.append(processed_text)
            stars.append(star)
            useful.append(use)
            funny.append(fun)
            cool.append(coo)

            counter += 1
            if counter == 100000:
                break
    return texts, stars, cool, useful, funny

# Read and preprocess reviews
texts, stars, cool, useful, funny = read_and_preprocess_reviews()

# Convert cool, useful, and funny votes to star ratings
max_cool = max(cool)
min_cool = min(cool)
cool_stars = [int((x - min_cool) / (max_cool - min_cool) * 4 + 1) for x in cool]

max_useful = max(useful)
min_useful = min(useful)
useful_stars = [int((x - min_useful) / (max_useful - min_useful) * 4 + 1) for x in useful]

max_funny = max(funny)
min_funny = min(funny)
funny_stars = [int((x - min_funny) / (max_funny - min_funny) * 4 + 1) for x in funny]

# Combine all star ratings into one list
all_stars = [list(x) for x in zip(stars, cool_stars, useful_stars, funny_stars)]

# Split the data into train, validation, and test sets (80-10-10 split)
X_train, X_test, y_train, y_test = train_test_split(texts, all_stars, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#Convert text to numerical vectors using Count Vectorization
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_val_vectors = vectorizer.transform(X_val)
X_test_vectors = vectorizer.transform(X_test)

# Define function to extract features from text using Count Vectorization
def extract_features(X_train, X_val, X_test, k):
    vectorizer = CountVectorizer(max_features=k)
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)
    return X_train, X_val, X_test, vectorizer

# Set k values for each category
k_stars = 26000
k_useful = 50
k_funny = 35000
k_cool = 35000

# Extract features using Count Vectorization and SelectKBest feature selection
X_train_stars, X_val_stars, X_test_stars, vectorizer_stars = extract_features(X_train, X_val, X_test, k_stars)
X_train_useful, X_val_useful, X_test_useful, vectorizer_useful = extract_features(X_train, X_val, X_test, k_useful)
X_train_funny, X_val_funny, X_test_funny, vectorizer_funny = extract_features(X_train, X_val, X_test, k_funny)
X_train_cool, X_val_cool, X_test_cool, vectorizer_cool = extract_features(X_train, X_val, X_test, k_cool)

# Select k best features using chi-squared test

selector_stars = SelectKBest(chi2,k=k_stars)
selector_useful = SelectKBest(chi2, k=k_useful)
selector_funny = SelectKBest(chi2, k=k_funny)
selector_cool = SelectKBest(chi2, k=k_cool)

X_train_stars = selector_stars.fit_transform(X_train_stars, [y[0] for y in y_train])
X_val_stars = selector_stars.transform(X_val_stars)
X_test_stars = selector_stars.transform(X_test_stars)

X_train_useful = selector_useful.fit_transform(X_train_useful, [y[2] for y in y_train])
X_val_useful = selector_useful.transform(X_val_useful)
X_test_useful = selector_useful.transform(X_test_useful)

X_train_funny = selector_funny.fit_transform(X_train_funny, [y[3] for y in y_train])
X_val_funny = selector_funny.transform(X_val_funny)
X_test_funny = selector_funny.transform(X_test_funny)

X_train_cool = selector_cool.fit_transform(X_train_cool, [y[1] for y in y_train])
X_val_cool = selector_cool.transform(X_val_cool)
X_test_cool = selector_cool.transform(X_test_cool)

#Train a decision tree classifier on the training set
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train, y_train)

   # Evaluate on validation set
   
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_cr = classification_report(y_val, y_val_pred)
    
    
    # Evaluate on test set
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_cr = classification_report(y_test, y_test_pred)
   
    return clf

clf_stars = train_and_evaluate(X_train_stars, [x[0] for x in y_train], X_val_stars, [x[0] for x in y_val], X_test_stars, [x[0] for x in y_test])
y_pred_stars = clf_stars.predict(X_test_stars)
print("Stars:")
print("Validation Accuracy:", accuracy_score([x[0] for x in y_val], clf_stars.predict(X_val_stars)))
print("Validation Classification Report:\n", classification_report([x[0] for x in y_val], clf_stars.predict(X_val_stars)))
print("Test Accuracy:", accuracy_score([x[0] for x in y_test], y_pred_stars))
print("Test Classification Report:\n", classification_report([x[0] for x in y_test], y_pred_stars))
print()

clf_useful = train_and_evaluate(X_train_useful, [x[2] for x in y_train], X_val_useful, [x[2] for x in y_val], X_test_useful, [x[2] for x in y_test])
y_pred_useful = clf_useful.predict(X_test_useful)
print("Useful:")
print("Validation Accuracy:", accuracy_score([x[2] for x in y_val], clf_useful.predict(X_val_useful)))
print("Validation Classification Report:\n", classification_report([x[2] for x in y_val], clf_useful.predict(X_val_useful)))
print("Test Accuracy:", accuracy_score([x[2] for x in y_test], y_pred_useful))
print("Test Classification Report:\n", classification_report([x[2] for x in y_test], y_pred_useful))
print()

clf_funny = train_and_evaluate(X_train_funny, [x[3] for x in y_train], X_val_funny, [x[3] for x in y_val], X_test_funny, [x[3] for x in y_test])
y_pred_funny = clf_funny.predict(X_test_funny)
print("Funny:")
print("Validation Accuracy:", accuracy_score([x[3] for x in y_val], clf_funny.predict(X_val_funny)))
print("Validation Classification Report:\n", classification_report([x[3] for x in y_val], clf_funny.predict(X_val_funny)))
print("Test Accuracy:", accuracy_score([x[3] for x in y_test], y_pred_funny))
print("Test Classification Report:\n", classification_report([x[3] for x in y_test], y_pred_funny))
print()

clf_cool = train_and_evaluate(X_train_cool, [x[1] for x in y_train], X_val_cool, [x[1] for x in y_val], X_test_cool, [x[1] for x in y_test])
y_pred_cool = clf_cool.predict(X_test_cool)
print("Cool:")
print("Validation Accuracy:", accuracy_score([x[1] for x in y_val], clf_cool.predict(X_val_cool)))
print("Validation Classification Report:\n", classification_report([x[1] for x in y_val], clf_cool.predict(X_val_cool)))
print("Test Accuracy:", accuracy_score([x[1] for x in y_test], y_pred_cool))
print("Test Classification Report:\n", classification_report([x[1] for x in y_test], y_pred_cool))



clf_stars = train_and_evaluate(X_train_stars, [x[0] for x in y_train], X_val_stars, [x[0] for x in y_val], X_test_stars, [x[0] for x in y_test])
y_pred_stars = clf_stars.predict(X_test_stars)
print("Stars:")
print("Test Accuracy:", accuracy_score([x[0] for x in y_test], y_pred_stars))
print("Test Classification Report:\n", classification_report([x[0] for x in y_test], y_pred_stars))
print()

clf_useful = train_and_evaluate(X_train_useful, [x[2] for x in y_train], X_val_useful, [x[2] for x in y_val], X_test_useful, [x[2] for x in y_test])
y_pred_useful = clf_useful.predict(X_test_useful)
print("Useful:")
print("Test Accuracy:", accuracy_score([x[2] for x in y_test], y_pred_useful))
print("Test Classification Report:\n", classification_report([x[2] for x in y_test], y_pred_useful))
print()

clf_funny = train_and_evaluate(X_train_funny, [x[3] for x in y_train], X_val_funny, [x[3] for x in y_val], X_test_funny, [x[3] for x in y_test])
y_pred_funny = clf_funny.predict(X_test_funny)
print("Funny:")
print("Test Accuracy:", accuracy_score([x[3] for x in y_test], y_pred_funny))
print("Test Classification Report:\n", classification_report([x[3] for x in y_test], y_pred_funny))
print()

clf_cool = train_and_evaluate(X_train_cool, [x[1] for x in y_train], X_val_cool, [x[1] for x in y_val], X_test_cool, [x[1] for x in y_test])
y_pred_cool = clf_cool.predict(X_test_cool)
print("Cool:")
print("Test Accuracy:", accuracy_score([x[1] for x in y_test], y_pred_cool))
print("Test Classification Report:\n", classification_report([x[1] for x in y_test], y_pred_cool))



#Save the classifier
with open('classifier.pkl', 'wb') as f:
 pickle.dump(clf, f)