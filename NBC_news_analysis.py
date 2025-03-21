# Importing python libraries
import os
import glob
import pypandoc
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob


# NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')


# Initialize PorterStemmer to perform stemming
ps = PorterStemmer()

# creating a set of stopwords to remove 
stop_words = set(stopwords.words('english'))

# Initialize BERT sentiment analysis components
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Load spaCy model and add TextBlob sentiment analyzer
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# This function reads .docx files using pypandoc
def read_docx(file_path):
    return pypandoc.convert_file(file_path, 'plain')

# This function reads .txt files
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# This function reads all documents in a directory with multiple folders consisting of .txt and .docx files
def read_documents(directory_path):
    documents = {}
    file_id = 1
    if os.path.isdir(directory_path):
        for file_path in glob.glob(os.path.join(directory_path, '*')):
            if file_path.endswith('.docx'):
                documents[file_id] = read_docx(file_path)
            elif file_path.endswith('.txt'):
                documents[file_id] = read_txt(file_path)
            file_id += 1
    return documents

# This function performs Tokenization, stopword removal, and stemming and returns a list of tokens
def process_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalnum()]
    filtered_tokens = [ps.stem(token) for token in tokens if token not in stop_words]
    return filtered_tokens

# This function builds the inverted index and positional index from a dictionary of document collection
def build_indexes(documents):
    inverted_index = defaultdict(set)
    positional_index = defaultdict(lambda: defaultdict(list))

    for doc_id, text in documents.items():
        tokens = word_tokenize(text)
        for position, token in enumerate(tokens):
            token = token.lower()
            if token.isalnum() and token not in stop_words:
                stemmed_token = ps.stem(token)
                inverted_index[stemmed_token].add(doc_id)
                positional_index[stemmed_token][doc_id].append(position)
    
    return inverted_index, positional_index

# This function generates the word cloud
def generate_word_cloud(tokens):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tokens))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# This function prints the first 10 tokens in the inverted index and positional index
def print_analysis(inverted_index, positional_index):
    total = 5
    i = 0
    j = 0
    print("Inverted Index: (First 5)")
    for term, doc_ids in inverted_index.items():
        if i < total:
            print(f"{term}: {sorted(doc_ids)}\n")
            i += 1
        else:
            break

    
    print("\nPositional Index: (First 5)")
    for term, postings in positional_index.items():
        if j < total:
            print(f"{term}: {dict(postings)}\n")
            j += 1
        else:
            break

# This function plots the term frequency using positional index for top 20 tokens
def plot_term_frequency(positional_index):
    term_freq = {term: sum(len(positions) for positions in doc_positions.values()) 
                 for term, doc_positions in positional_index.items()}
    common_terms = Counter(term_freq).most_common(20)
    terms, frequencies = zip(*common_terms)
    
    plt.figure(figsize=(12, 6))
    plt.bar(terms, frequencies)
    plt.xticks(rotation=45)
    plt.xlabel('Terms')
    plt.ylabel('Frequencies')
    plt.title('Top 20 Most Frequent Terms')
    plt.show()

# This function plots the word co-occurrence network for 20 most common words
def plot_word_cooccurrence(tokens):
    bigrams = list(nltk.bigrams(tokens))
    bigram_freq = Counter(bigrams)
    common_bigrams = bigram_freq.most_common(20)
    
    G = nx.Graph()
    for (word1, word2), freq in common_bigrams:
        G.add_edge(word1, word2, weight=freq)
    
    # Plotting the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2)
    nx.draw(G, pos, with_labels=False, node_size=30, font_size=10, edge_color='#AAAAAA')
    
    # Writing the words corresponding to each node beneath the node 
    label_pos = {node: (x, y - 0.05) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=10)
    
    plt.title('Word Co-occurrence Network')
    plt.show()

# Perform sentiment analysis using BERT
def bert_sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    scores = outputs[0][0].detach().numpy()
    scores = torch.nn.functional.softmax(torch.tensor(scores), dim=0)
    return scores

# Plot sentiment scores
def plot_sentiment_scores(sentiment_scores, title):
    labels = ['negative', 'mostly negative', 'neutral', 'mostly positive', 'positive']
    plt.figure(figsize=(10, 5))
    plt.bar(labels, sentiment_scores, color=['red', 'orange', 'yellow', 'green', 'blue'])
    plt.xlabel('Sentiment')
    plt.ylabel('Scores')
    plt.title(title)
    plt.show()

def spacy_sentiment_analysis(text):
    doc = nlp(text)
    return doc._.polarity, doc._.subjectivity

def spacy_classifier(documents):
    sentiments = {}
    sentiments[-1] = [] #Negative Sentiment
    sentiments[0] = [] #Neutral Sentiment
    sentiments[1] = [] #Positive Sentiment
    data_type = {}
    data_type[0] = [] #Factual Data
    data_type[1] = [] #Subjective Data

    for i, j in documents.items():
        polarity, subjectivity = spacy_sentiment_analysis(j)
        if polarity < 0.0000:
            sentiments[-1].append((i,polarity))
        if polarity > 0.0000 and polarity < 0.1000:
            sentiments[0].append((i,polarity))
        if polarity > 0.1000:
            sentiments[1].append((i,polarity))
        
        if subjectivity < 0.5000:
            data_type[0].append((i,subjectivity))
        if subjectivity > 0.5000:
            data_type[1].append((i,subjectivity))
        
        if i == 1:
            print('\nspaCy Sentiment Analysis Scores for first 10 documents - \n')
        if i < 11:
            print(f"spaCy Sentiment Analysis Scores for document {i} :")
            print(f"Polarity: {polarity:.4f}")
            print(f"Subjectivity: {subjectivity:.4f}\n")

    return sentiments, data_type

def plot_spacy_sentiment(sentiments, data_type):
    neg = len(sentiments[-1])
    neu = len(sentiments[0])
    pos = len(sentiments[1])
    total = neg + neu + pos
    scores = [neg/total , neu/total , pos/total] 
    print(f'spaCy Sentiment Scores:\nNegative Sentiment: {scores[0]}\nNeutral Sentiment: {scores[1]}\nPositive Sentiment: {scores[2]} ')
    labels = ['negative', 'neutral', 'positive']
    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores , color=['red','yellow', 'blue'])
    plt.xlabel('Sentiment')
    plt.ylabel('Scores')
    plt.title('spaCy Sentiment Analysis')
    plt.show()

    factual = len(data_type[0])
    subjective = len(data_type[1])
    labels = ['Factual', 'Subjective']
    print(f'\nDocument Types:\nFactual Documents: {factual}\nSubjective Documents: {subjective}')
    plt.figure(figsize=(10, 5))
    plt.bar(labels, [factual, subjective] , color=['yellow', 'blue'])
    plt.xlabel('Data Type')
    plt.ylabel('Count')
    plt.title('spaCy Data Analysis')
    plt.show()


# Main function
def main():
    # Location of the data directory on my system
    directory_path = '/Users/venkatesh/Documents/test_project_dir/Information_Retrieval(ISTE-612)/Project/NBC_Data'
    # Reading documents from the directory
    documents = read_documents(directory_path)
    
    # Processing documents 
    all_tokens = []
    for doc_id, text in documents.items():
        tokens = process_text(text)
        all_tokens.extend(tokens)
    
    #Building the indexes
    inverted_index, positional_index = build_indexes(documents)

    # Print analysis
    print_analysis(inverted_index, positional_index)

    # Generate word cloud
    generate_word_cloud(all_tokens)

    # Term Frequency Analysis
    plot_term_frequency(positional_index)
    
    # Word Co-occurrence Network
    plot_word_cooccurrence(all_tokens)

    # Sentiment Analysis
    full_text = ' '.join(documents.values())
    # BERT Sentiment Analysis
    bert_scores = bert_sentiment_analysis(full_text)
    print("\nBERT Sentiment Analysis Scores:")
    for i, score in enumerate(bert_scores):
        print(f"{i+1} stars: {score:.4f}")
    plot_sentiment_scores(bert_scores, 'BERT Sentiment Analysis')
    
    sentiments, data_type = spacy_classifier(documents)
    plot_spacy_sentiment(sentiments, data_type)
    

if __name__ == "__main__":
    main()