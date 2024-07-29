import re

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def load_data(filename, colname):
    """
    Read in input file and load data

    filename: csv file
    colname: column name for texts
    return: X and y dataframe
    """
    print('************** Loading Data ************\n')
    ## Read in data from input file

    df = pd.read_csv(filename, header=0, delimiter="\t", quoting=3)
    print("\nSummary of data\n")
    df.info()

    # Check number of rows and columns
    print("No of Rows: {}".format(df.shape[0]))
    print("No of Columns: {}".format(df.shape[1]))

    # Check the Last 3 instances
    print("\nData View: Last 3 Instances\n")
    print(df.tail(3))

    # Check label class
    print('\nClass Counts(label, row): Total')
    print(df[colname[-1]].value_counts())
    #### Check the first 5 instances
    print("\nData View: First 5 Instances\n")
    print(df.head(5))

    #### Remove duplicates using 'id' column and keep first occurrence
    df.drop_duplicates(subset=['id'], keep='first', inplace=True)

    #### Check number of rows and columns
    print("No of Rows(After removing duplicates): {}".format(df.shape[0]))

    #### extract col
    df = df[colname]

    ## Split into X and y (target)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    return X, y


def split_data(X, y):
    print("\n************** Spliting Data **************\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    ## Check the data view of each data set

    print("\n************** Data After Splitting **************\n")

    ## Data Shape
    print("Train Data: {}".format(X_train.shape))
    print("Test Data: {}".format(X_test.shape))

    ## Label Distribution
    print('\nClass Counts(label, row): Train')
    print(y_train.value_counts())
    print('\nClass Counts(label, row): Test')
    print(y_test.value_counts())

    ## Display the first 5 instances of X data
    print("\nFirst 5 Instance: Train")
    print(X_train.head())
    print("\nFirst 5 Instance: Test")
    print(X_test.head())

    ## Reset index

    print("\n************** Resetting Index **************\n")

    # Train Data
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # Test Data
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    ## Check data

    print("\n************** Data After Resetting **************\n")

    ## Display the first 5 instances of X data
    print("\nFirst 5 Instance: Train\n")
    print(X_train.head())
    print("\nFirst 5 Instance: Test\n")
    print(X_test.head())

    return (X_train, X_test, y_train, y_test)


# Convert sentences into word lists
def preprocess_data(X_data_raw):
    # 1. Remove HTML
    X_data = BeautifulSoup(X_data_raw, "html.parser").get_text()

    # 2. Remove punctuation
    X_data = re.sub('[^\w\s]', '', X_data)

    # 3. Remove non-letters
    X_data = re.sub("[^a-zA-Z]", " ", X_data)

    # 4. Convert to lower case
    X_data = X_data.lower()

    # 5. tokenize sentence
    X_data = nltk.word_tokenize(X_data)

    # 6. remove stopwords
    stopword_list = stopwords.words("english")
    X_data = [word for word in X_data if word not in stopword_list]

    # 7. stemming
    stemmer = PorterStemmer()
    X_data = [stemmer.stem(y) for y in X_data]

    # 8. get str
    # X_data = " ".join(X_data)

    return X_data
