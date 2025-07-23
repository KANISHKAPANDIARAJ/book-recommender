import pandas as pd

# Load datasets
books = pd.read_csv('books.csv')
tags = pd.read_csv('tags.csv')
book_tags = pd.read_csv('book_tags.csv')

# Print columns of each to check their headers
print("books.csv columns:", books.columns.tolist())
print("tags.csv columns:", tags.columns.tolist())
print("book_tags.csv columns:", book_tags.columns.tolist())

# Print first few rows to sample data
print("\nbook_tags.csv sample rows:")
print(book_tags.head())
