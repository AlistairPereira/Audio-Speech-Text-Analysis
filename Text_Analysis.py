
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import spacy
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# Initialize logging for gensim warnings
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

# Initialize NLTK for Sentiment Analysis
nltk.download('vader_lexicon')
nltk.download('punkt')  # Download punkt tokenizer
nltk.download('stopwords')  # Download stopwords if needed
nltk.download('punkt_tab')  # In case 'punkt' alone does not fix the issue


# Load spaCy model for lemmatization and stop words
nlp = spacy.load('en_core_web_sm')

# Read text from the file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Sentiment Analysis
def perform_sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    # Visualization
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(sentiment_scores.keys()), y=list(sentiment_scores.values()), palette='coolwarm')
    plt.title('Sentiment Analysis Scores')
    plt.xlabel('Sentiment')
    plt.ylabel('Score')
    plt.show()
    
    return sentiment_scores

# Topic Modelling

def perform_topic_modeling(text, num_topics=3):
    # Preprocess the text using spaCy
    doc = nlp(text)
    cleaned_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    
    # Vectorize the cleaned text
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([cleaned_text])
    
    # Fit the LDA model
    lda = LDA(n_components=num_topics, random_state=42)
    lda.fit(X)
    
    # Get the topic-word distributions
    topics = lda.components_
    feature_names = vectorizer.get_feature_names_out()
    topic_words = {}
    
    # Get topic distribution for the document
    topic_distribution = lda.transform(X)  # Proportion of each topic in the document
    
    # Print the topic distribution
    print("\nTopic Distribution in Document:")
    for i, doc_topic_dist in enumerate(topic_distribution):
        print(f"\nDocument {i + 1} topic distribution:")
        for topic_idx, dist in enumerate(doc_topic_dist):
            print(f"  Topic {topic_idx + 1}: {dist * 100:.2f}%")
    
    # Create a dictionary for the top words for each topic
    for topic_idx, topic in enumerate(topics):
        topic_words[topic_idx] = [feature_names[i] for i in topic.argsort()[:-6 - 1:-1]]
        
        # Display the top words for each topic
        print(f"\nTop words for Topic {topic_idx + 1}: {', '.join(topic_words[topic_idx])}")
        
        # Visualize topic distribution (Bar Chart)
        if topic_idx == 0 or topic_idx == 1:  # Only visualize Topic 1 and Topic 2
            plt.figure(figsize=(8, 4))
            sns.barplot(x=topic_words[topic_idx], y=topic[topic.argsort()[:-6 - 1:-1]], palette='viridis')
            plt.title(f'Topic {topic_idx + 1} Top Words')
            plt.xticks(rotation=30)
            plt.ylabel('Importance')
            plt.show()
            
            # Generate Word Cloud for the specific topic
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(topic_words[topic_idx]))
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Word Cloud for Topic {topic_idx + 1}')
            plt.axis('off')
            plt.show()
    
    return topic_words

#Text Summarization


def summarize_text(text, num_sentences=3):
    # Step 1: Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Step 2: Compute the TF-IDF matrix for the sentences
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    # Step 3: Compute the cosine similarity between all sentence pairs
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Step 4: Rank the sentences based on the sum of cosine similarities
    sentence_scores = cosine_sim.sum(axis=1)
    
    # Step 5: Select the top `num_sentences` sentences with the highest scores
    ranked_sentences_idx = sentence_scores.argsort()[-3:][::-1]
    
    # Create the summary by selecting the top-ranked sentences
    summary = [sentences[idx] for idx in ranked_sentences_idx]
    
    # Visualization 1: Bar Chart of Sentence Scores
    plt.figure(figsize=(12, 6))
    sns.barplot(x=np.arange(len(sentences)), y=sentence_scores, palette='viridis')
    plt.axhline(y=np.mean(sentence_scores), color='red', linestyle='--', label='Average Score')
    plt.xticks(ticks=np.arange(len(sentences)), labels=[f'Sentence {i+1}' for i in range(len(sentences))], rotation=45)
    plt.xlabel('Sentences')
    plt.ylabel('Sentence Importance Score')
    plt.title('Sentence Importance Scores Based on TF-IDF & Cosine Similarity')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
    return ' '.join(summary)


def main():
    file_path = 'converted_data.txt'  

    transcription_text = read_text_file(file_path)
    
    # Perform Sentiment Analysis
    sentiment_result = perform_sentiment_analysis(transcription_text)
    print("\nSentiment Analysis:\n", sentiment_result)

    # Perform Topic Modeling
    topics_result = perform_topic_modeling(transcription_text)
    print("\nTopic Modeling:\n", topics_result)
    
     # Perform Text Summarization
    summary = summarize_text(transcription_text, num_sentences=3)
    print("\nText Summarization:\n", summary)

if __name__ == "__main__":
    main()
