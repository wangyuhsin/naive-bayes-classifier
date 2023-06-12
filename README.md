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
# Import the required modules
from bayes import NaiveBayes621
from sklearn.feature_extraction.text import CountVectorizer

# Create an instance of the NaiveBayes621 class
nb = NaiveBayes621()

# Generate sample data
documents = [
    "This is a positive document",
    "This is a negative document",
    "Another positive document",
    "Another negative document"
]
labels = [1, 0, 1, 0]

# Create a vectorizer to convert text to word vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents).toarray()

# Fit the classifier to the data
nb.fit(X, labels)

# Predict the class of a new document
new_document = ["This is a test document"]
X_new = vectorizer.transform(new_document).toarray()
prediction = nb.predict(X_new)
print(prediction)  # Output: [1]
```

In this example, we create a `NaiveBayes621` instance, generate sample data consisting of four documents with their corresponding labels, and convert the documents into word vectors using a `CountVectorizer`. We then fit the classifier to the data and predict the class of a new test document.

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

The initial codebase and project structure is adapted from the MSDS 621 course materials provided by the University of San Francisco (USFCA-MSDS). Special thanks to the course instructors for the inspiration.
