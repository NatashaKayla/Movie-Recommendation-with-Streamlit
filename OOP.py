# Class Group 1 
# Members :
# - Natasha Kayla Cahyadi - 2702235891
# - Jeremy Djohar Riyadi - 2702219572
# Class : LC09 - Model Deployment

# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl

class Preprocessor:
    # Initialize the Preprocessor with the path to the dataset file
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.smd = None
        self.categorical_columns = None
        self.numerical_columns = None

    # Read the CSV data from the given filepath into a DataFrame
    def read_data(self):
        self.df = pd.read_csv(self.filepath)
        return self.df

    # Filter the dataset to keep only rows where the 'type' column equals 'Movie'
    def filter_movie(self):
        self.df = self.df[self.df['type'] == 'Movie']

    # Remove the 'type' column from the DataFrame
    def drop_column(self):
        self.df.drop(columns=['type'], inplace=True)

    # Remove the 'show_id' column which acts as an identifier and may not be useful for analysis
    def drop_identifier(self):
        self.df.drop(columns=['show_id'], inplace=True)

    # Fill all missing values in the DataFrame with the string 'Unknown'
    def handle_missing_values(self):
        self.df.fillna('Unknown', inplace=True)

    # Convert the 'date_added' column to datetime format, coercing errors to NaT (Not a Time)
    def convert_to_datetime(self):
        self.df['date_added'] = pd.to_datetime(self.df['date_added'], errors='coerce')

    # Replace 'UR' and 'Unknown' ratings with 'NR' to standardize rating values
    def change_to_nr(self):
        self.df.loc[self.df['rating'].isin(['UR', 'Unknown']), 'rating'] = 'NR'

    # Remove rows with anomalous rating values that appear to be durations instead of ratings
    def delete_anomalies(self):
        self.df = self.df[~self.df['rating'].isin(['74 min', '66 min', '84 min'])]

    # Create a copy of the cleaned DataFrame for further processing or analysis
    def create_working_copy(self):
        self.smd = self.df.copy()
        return self.smd

# Class Modelling
class Modeling:
    # Initialize the recommender with a DataFrame smd and set up variables and TF-IDF vectorizer
    def __init__(self, smd):
        self.smd = smd
        self.x = None
        self.tfidf_matrix = None
        self.cosine_sim = None

        # Configure TF-IDF Vectorizer with English stop words, bigrams, and term frequency filtering
        self.model =TfidfVectorizer(
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.8,
                    sublinear_tf=True)
        
    # Create a combined 'text' column by concatenating title, director (repeated twice), cast (thrice), country,rating, listed_in (five times), and description (twice), then convert to lowercase and drop missing texts
    def build_text_column(self):
        self.smd['text'] = self.smd['title'] + ', ' + self.smd['director'] * 2 + ', ' + self.smd['cast'] * 3 + ', ' + self.smd['country'] + ', ' +  self.smd['rating'] + ', ' + self.smd['listed_in'] * 5 + ', ' + self.smd['description'] * 2
        self.smd.dropna(subset=['text'], inplace=True)
        self.smd['text'] = self.smd['text'].str.lower()

    # Save the smd DataFrame to a pickle file named 'smd.pkl'
    def build_smd_pickle(self):
        pkl.dump(self.smd, open('smd.pkl', 'wb'))
    
    # Make a copy of smd and remove duplicate rows based on the 'text' column, resetting the index
    def duplicate_text_column(self):
        self.x = self.smd.copy()
        self.x = self.x.drop_duplicates(subset=['text']).reset_index(drop=True)

    # Fit the TF-IDF model on the 'text' column of x, convert it to an array, and compute cosine similarity matrix using linear kernel
    def fit_model(self):
        self.tfidf_matrix = self.model.fit_transform(self.x['text']).toarray()
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    # Save the TF-IDF matrix to a pickle file named 'tfidf_matrixx.pkl'
    def build_model_pickle(self):
        pkl.dump(self.tfidf_matrix, open('tfidf_matrixx.pkl', 'wb'))

    # Save the DataFrame x (with unique texts) to a pickle file named 'x.pkl'
    def build_cosine_sim_pickle(self):
        pkl.dump(self.x, open('x.pkl', 'wb'))

    # Retrieve top similar movies/shows to the given title based on cosine similarity scores
    def get_recommendations(self, title, num_recommend=5):
        try:
            # Create a Series mapping titles to their indices
            indices = pd.Series(self.x.index, index=self.x['title']).drop_duplicates()
            idx = indices[title]
            # Handle case if multiple entries exist for the title by taking the first
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            # Get similarity scores for the given index and sort them in descending order
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Select top similar items excluding the input title itself
            top_similar = sim_scores[1:num_recommend+1]
            movie_indices = [i[0] for i in top_similar]
            # Return recommended items with similarity score, dropping the 'text' column
            ret_smd = self.smd.iloc[movie_indices].copy()
            ret_smd['Score'] = [i[1] for i in top_similar]
            return ret_smd.drop(columns=['text'], errors='ignore')
        except KeyError:
            # Handle case when the title is not found in the dataset
            print(f"{title} not found. Please check the title or try another.")
            return pd.DataFrame()

# Initialize the preprocessing pipeline with the Netflix dataset
preprocessor = Preprocessor('netflix_titles.csv')

# Run all preprocessing steps in sequence
preprocessor.read_data()
preprocessor.filter_movie()
preprocessor.drop_column()
preprocessor.drop_identifier()
preprocessor.handle_missing_values()
preprocessor.convert_to_datetime()
preprocessor.change_to_nr()
preprocessor.delete_anomalies()
preprocessor.create_working_copy()

# Initialize the recommendation model using the cleaned data
modeling = Modeling(preprocessor.smd)
# Prepare the text column for TF-IDF modeling
modeling.build_text_column()
# Save the prepared dataset into a pickle file
modeling.build_smd_pickle()
# Remove duplicate text entries
modeling.duplicate_text_column()
# Fit TF-IDF model and compute similarity matrix
modeling.fit_model()
# Save model and processed data into pickle files
modeling.build_model_pickle()
modeling.build_cosine_sim_pickle()

# Print the recommendations
recommendations = modeling.get_recommendations('Naruto Shippuden: The Movie')
print(recommendations)