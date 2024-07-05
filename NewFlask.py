from flask import Flask, request, jsonify
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Membaca file JSON
with open('C:\\Users\\MSI PC\\Desktop\\Project-TA\\combined_resep.json', 'r', encoding='utf-8') as file:
    recipes = json.load(file)


# Menyiapkan data dan label (label harus sudah ada dalam dataset)
data = []
labels = []
for recipe in recipes:
    title = recipe.get('Title', '')
    ingredients = recipe.get('Ingredients', '')
    steps = recipe.get('Steps', '')
    combined_text = ingredients + " " + steps  # Menggabungkan bahan dan langkah-langkah
    data.append(combined_text)
    labels.append(recipe.get('Category', 'other'))  # Pastikan ada field 'Category' dalam dataset

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data)

# Split data menjadi train dan test set
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)

# Hyperparameter tuning menggunakan GridSearchCV
param_grid = {'n_neighbors': range(1, 31)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Model terbaik
best_knn = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluasi model
y_pred = best_knn.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=best_knn.classes_, yticklabels=best_knn.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# Akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi: {accuracy * 100:.2f}%")

@app.route('/recommend', methods=['POST'])
def recommend():
    input_data = request.json
    input_ingredients = input_data.get('ingredients', '')

    input_ingredients_list = [ingredient.strip() for ingredient in input_ingredients.split(',')]
    input_tfidf = vectorizer.transform([' '.join(input_ingredients_list)])
    predicted_category = best_knn.predict(input_tfidf)[0]

    selected_recipes = [recipe for recipe, label in zip(recipes, labels) if label == predicted_category]

    if not selected_recipes:
        return jsonify([])

    selected_contents = [recipe['Ingredients'] + " " + recipe['Steps'] for recipe in selected_recipes]
    selected_tfidf_matrix = vectorizer.transform(selected_contents)
    cosine_similarities = cosine_similarity(input_tfidf, selected_tfidf_matrix)
    
    threshold = 0.1  # Nilai threshold cosine similarity yang ditetapkan
    similarity_scores = cosine_similarities[0]
    top_indices = [i for i in range(len(similarity_scores)) if similarity_scores[i] >= threshold]
    top_indices = sorted(top_indices, key=lambda i: similarity_scores[i], reverse=True)[:5]

    recommendations = []
    for index in top_indices:
        recipe = selected_recipes[index]
        recommendations.append({
            'title': recipe['Title'],
            'ingredients': recipe['Ingredients'],
            'steps': recipe['Steps']
        })

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)