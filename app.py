from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load CSVs
books = pd.read_csv('books.csv')
tags = pd.read_csv('tags.csv')
book_tags = pd.read_csv('book_tags.csv')
full_desc = pd.read_csv('full_book_descriptions.csv')

# Merge tags to get genre names
book_tag_names = pd.merge(book_tags, tags, on='tag_id', how='left')
book_tags_grouped = (
    book_tag_names
    .groupby('goodreads_book_id')['tag_name']
    .apply(lambda x: ', '.join(x.unique()))
    .reset_index()
)

# Merge genres into books DataFrame
books = pd.merge(
    books,
    book_tags_grouped,
    left_on='book_id',
    right_on='goodreads_book_id',
    how='left'
)

# Merge full descriptions into books DataFrame by book_id
books = pd.merge(
    books,
    full_desc[['book_id', 'description']],
    on='book_id',
    how='left'
)

# Rename columns for clarity
books.rename(columns={'tag_name': 'genres'}, inplace=True)

# Drop redundant column
books.drop(columns=['goodreads_book_id'], inplace=True, errors='ignore')

# Create combined description for TF-IDF (title + author + full description)
books['desc'] = (
    books['title'].fillna('') + ' ' +
    books['authors'].fillna('') + ' ' +
    books['description'].fillna('')
)

# Prepare TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['desc'])


def get_recommendations(title, top_n=5):
    matched_books = books[books['title'].str.contains(title, case=False, na=False)]
    if matched_books.empty:
        return []
    idx = matched_books.index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n - 1:-1][::-1]
    recommendations = books.iloc[similar_indices][['title', 'authors', 'description']]
    return recommendations.to_dict(orient='records')


def search_by_genre(genre):
    if 'genres' not in books.columns:
        return []
    filtered_books = books[books['genres'].str.contains(genre, case=False, na=False)]
    return filtered_books[['title', 'authors', 'genres', 'description']].to_dict(orient='records')


@app.route('/')
def home():
    top_books = books.sample(10)[['title', 'authors']].to_dict(orient='records')
    return render_template('index.html', books=top_books)


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    recommendations = []
    if request.method == 'POST':
        user_input = request.form['book_title']
        recommendations = get_recommendations(user_input)
    return render_template('recommend.html', recommendations=recommendations)


@app.route('/genre', methods=['POST'])
def genre():
    genre = request.form['genre']
    results = search_by_genre(genre)
    return render_template('genre_results.html', genre=genre, books=results)


if __name__ == '__main__':
    app.run(debug=True)
