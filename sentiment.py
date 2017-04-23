import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])


def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg


def formDictionary(dictionary, dataset):
    for words in dataset:
        set_data = set(words)
        for word in set_data:																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																			
            if not word in dictionary:
            	dictionary[word] = 0
            dictionary[word] += 1
    return dictionary

def constructVector(dataset, wordlist):
	result = []
	for text in dataset:
		temp = []
		for t in wordlist:
			if t in text:
				temp.append(1)
			else:
				temp.append(0)
		result.append(temp)
	return result

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    

    positive_word_list, negative_word_list = {}, {}
    
    positive_word_list = formDictionary(positive_word_list, train_pos)
    positive_word_list = formDictionary(positive_word_list, test_pos)

    negative_word_list = formDictionary(negative_word_list, train_neg)
    negative_word_list = formDictionary(negative_word_list, test_neg)    
    
    train_pos_length = len(train_pos)
    test_pos_length = len(test_pos)
    train_neg_length = len(train_neg)
    test_neg_length = len(test_neg)

    positive_threshold = 0.01 * (train_pos_length + test_pos_length)
    negative_threshold = 0.01 * (train_neg_length + test_neg_length)
    
    all_words = set(positive_word_list.keys() + negative_word_list.keys())
    
    words_list = []

    for word in all_words:
        if not positive_word_list.has_key(word):
            positive_word_list[word] = 0
        if not negative_word_list.has_key(word):
            negative_word_list[word] = 0
        if word not in stopwords:
            if positive_word_list[word] >= positive_threshold or negative_word_list[word] >= negative_threshold:
                if positive_word_list[word] >= 2 * negative_word_list[word] or negative_word_list[word] >= 2 * positive_word_list[word]:
                   words_list.append(word)												

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    

    words_list = set(words_list)
    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec  = [], [], [], []
	
    train_pos_vec = constructVector(train_pos,words_list)
    train_neg_vec = constructVector(train_neg,words_list)
    test_pos_vec = constructVector(test_pos,words_list)
    test_neg_vec = constructVector(test_neg,words_list)
    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def label_sets(dataset,label):
    i = 0
    label_set = []
    for text in dataset:
        lab = [label + '_%s' % i]
        label_set.append(LabeledSentence(text,lab))
        i = i + 1
    return label_set

def get_features(model,dataset,label):
    i = 0
    vector = []
    for text in dataset:
        lab = label + '_%s' % i
        vector.append(model.docvecs[lab])
        i = i + 1
    return vector


def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    	
    labeled_train_pos, labeled_train_neg, labeled_test_pos, labeled_test_neg = [],[],[],[]    
    labeled_train_pos = label_sets(train_pos,'TRAIN_POS')
    labeled_train_neg = label_sets(train_neg,'TRAIN_NEG')
    labeled_test_pos = label_sets(test_pos,'TEST_POS')
    labeled_test_neg = label_sets(test_neg,'TEST_NEG')

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    
    
    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = [], [], [], []
    
    train_pos_vec = get_features(model,train_pos,"TRAIN_POS")
    train_neg_vec = get_features(model,train_neg,"TRAIN_NEG")
    test_pos_vec = get_features(model,test_pos,"TEST_POS")
    test_neg_vec = get_features(model,test_neg,"TEST_NEG")

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    
    
    X = train_pos_vec + train_neg_vec
 
    nb_model = sklearn.naive_bayes.BernoulliNB(alpha = 1.0, binarize = None)
    nb_model.fit(X,Y)
    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(X,Y)

    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    

    X = train_pos_vec + train_neg_vec

    nb_model = sklearn.naive_bayes.GaussianNB()
    lr_model = sklearn.linear_model.LogisticRegression()

    nb_model.fit(X,Y)
    lr_model.fit(X,Y)

    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    
    
    positive = model.predict(test_pos_vec)
    negative = model.predict(test_neg_vec)

    tp = list(positive).count("pos")
    fn = list(positive).count("neg")
    tn = list(negative).count("neg")
    fp = list(negative).count("pos")
    
    accuracy = float(tp + tn)/(tp+fp+tn+fn)

    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
