# Ai-powered-food-recipe-recommendation-system-

# Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading recipe dataset
recipes = pd.read_csv('recipes.csv')

# Creating TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fitting vectorizer to recipe ingredients
recipe_vectors = vectorizer.fit_transform(recipes['ingredients'])

# Function to get recommendations
def get_recommendations(ingredients, num_recipes=5):
  # Creating TF-IDF vector for input ingredients
  input_vector = vectorizer.transform([ingredients])
  
  # Calculating cosine similarity with recipe vectors
  similarities = cosine_similarity(input_vector, recipe_vectors).flatten()
  
  # Getting top N similar recipes
  top_indices = similarities.argsort()[-num_recipes:][::-1]
  return recipes.iloc[top_indices]

# Testing the function
ingredients = 'chicken, rice, vegetables'
recommendations = get_recommendations(ingredients)
print(recommendations[['name', 'ingredients', 'instructions']])
