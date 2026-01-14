import pandas as pd

def retrieveBook(index):
  # Read the books CSV file
  books_df = pd.read_csv('dataset\db_books.csv')
  book = books_df.iloc[index]

  # Read the stories CSV file
  stories_df = pd.read_csv('dataset\stories.csv')

  # Get the matching story at the same index
  story = stories_df.iloc[index]

  return {'book': book.to_dict(), 'story': story.to_dict()}