# Naive Bayes Classifier

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains a Python implementation of the Naive Bayes classifier. The classifier is trained on a collection of documents and can predict the class of new documents based on their word features.

## Introduction to Naive Bayes

Naive Bayes is a probabilistic classifier that is based on Bayes' theorem. It is called "naive" because it makes a strong assumption of feature independence, which is often considered oversimplified or "naive" in many real-world scenarios. Despite its simplifying assumption, Naive Bayes can still perform well in practice, often achieving good results in many classification tasks. Its simplicity and computational efficiency make it attractive, especially when dealing with large-scale data sets or real-time applications. 

Naive Bayes is commonly used for text classification tasks, such as spam filtering, sentiment analysis, and topic classification. It assumes that the features (words) are conditionally independent given the class label. The algorithm estimates two probabilities: the prior probability of each class (`p(c)`) and the likelihood of each word given the class (`P(w|c)`). It then uses these probabilities to calculate the posterior probability of each class given the document and predicts the class with the highest posterior probability.

However, it's important to note that Naive Bayes may not always be the best choice for every classification problem. Its assumption of feature independence may be strongly violated in certain domains, and it may struggle with capturing complex relationships between features. In such cases, more advanced models like decision trees, random forests, or deep learning architectures may be more appropriate.

The NaiveBayes621 class in the `bayes.py` file implements a binary Naive Bayes classifier. It can be trained on a matrix of word features and corresponding class labels, and then used to predict the class of new documents based on their word features. The classifier estimates the prior probability of each class and the likelihood of each word given each class. During prediction, it calculates the posterior probability of each class for a new document and assigns the document to the class with the highest probability.

## Functions

The `bayes.py` file also contains several utility functions that are used by the NaiveBayes621 class:

- `defaultintdict`: A subclass of `dict` that behaves like `defaultdict(int)` but does not add missing keys to the dictionary.
- `filelist(root)`: Returns a sorted list of fully-qualified filenames under a given root directory.
- `get_text(filename)`: Loads and returns the text of a text file.
- `words(text)`: Processes a string to extract a list of words, removing punctuation, numbers, and stop words.
- `load_docs(docs_dirname)`: Loads all text files under a given directory and returns a list of word lists, one per document.
- `vocab(neg, pos)`: Constructs a mapping from words to word indices based on the word lists of negative and positive documents.
- `vectorize(V, docwords)`: Returns a row vector that represents a document's word features based on a given word-index mapping.
- `vectorize_docs(docs, V)`: Returns a matrix where each row represents a document's word features.
- `kfold_CV(model, X, y, k)`: Performs k-fold cross-validation using a given model and word feature matrix and returns the accuracies of each fold.

Please refer to the inline comments in the `bayes.py` file for more detailed explanations of each function.

## Usage

To use the Naive Bayes classifier, follow these steps:

1. Import the `NaiveBayes621` class from the `bayes` module:
```python
from bayes import NaiveBayes621
```

2. Create an instance of the `NaiveBayes621` class:
```python
nb = NaiveBayes621()
```

3. Fit the classifier to your training data using the `fit(X, y)` method, where `X` is a 2D word vector matrix and `y` is a binary vector indicating the class of each document:
```python
nb.fit(X_train, y_train)
```

4. Predict the classes of new documents using the `predict(X)` method, where `X` is a 2D word vector matrix:
```python
predictions = nb.predict(X_test)
```

## Example

Here is an example of how to use the Naive Bayes classifier with sample data:

```python
from bayes import *

# Create an instance of the NaiveBayes621 class
nb = NaiveBayes621()

# Directory for negative and positive review files
neg_dir = 'review_polarity/txt_sentoken/neg/'
pos_dir = 'review_polarity/txt_sentoken/pos/'

# Load negative and positive review documents
neg = load_docs(neg_dir)
pos = load_docs(pos_dir)

# Create a vocabulary from the negative and positive review documents
V = vocab(neg, pos)

# Vectorize the negative and positive review documents using the vocabulary
vneg = vectorize_docs(neg, V)
vpos = vectorize_docs(pos, V)

# Create the feature matrix by stacking the negative and positive vectors
X = np.vstack([vneg, vpos])
# Create the target labels (0 for negative, 1 for positive)
y = np.vstack([np.zeros(shape=(len(vneg), 1)),
               np.ones(shape=(len(vpos), 1))]).reshape(-1)

# Train the Naive Bayes model with the feature matrix and target labels
nb.fit(X, y)

# Convert test instances into a numerical feature vector
test1 = np.expand_dims(vectorize(V, words("mostly very funny , the story is quite appealing.")), axis=0)
test2 = np.expand_dims(vectorize(V, words("there is already a candidate for the worst of 1997.")), axis=0)

# Make a prediction for test instances
prediction1 = nb.predict(test1)    # Output: array([1.])
prediction2 = nb.predict(test2)    # Output: array([0.])
```

The example performs sentiment analysis using the Naive Bayes algorithm. It starts by loading negative and positive review documents and creating a vocabulary based on them. The documents are then vectorized using the vocabulary, and a feature matrix and corresponding target labels are created. The Naive Bayes model is trained on this data. Finally, two test instances are converted into feature vectors and used to make predictions on their sentiment using the trained model.

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

The initial codebase and project structure is adapted from the MSDS 621 course materials provided by the University of San Francisco (USFCA-MSDS). Special thanks to the course instructors for the inspiration.
