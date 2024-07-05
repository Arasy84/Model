import os
import pandas as pd
import json
import re

folder_path = r'C:\Users\MSI PC\Desktop\Project-TA\TA\Resep'

def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9/,]', ' ', str(text))
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

dfs = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        # Mengambil kategori dari nama file
        category = filename.split('.')[0]  # Menggunakan nama file sebagai kategori, bisa dimodifikasi sesuai kebutuhan

        df = pd.read_csv(file_path, nrows=50)

        df['Title'] = df['Title'].apply(clean_text)
        df['Ingredients'] = df['Ingredients'].apply(clean_text)
        df['Steps'] = df['Steps'].apply(clean_text)
        
        df['Category'] = category  # Menambahkan kolom kategori
        
        dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

selected_columns = ["Title", "Ingredients", "Steps", "Category"]

selected_df = combined_df[selected_columns]

selected_dict = selected_df.to_dict(orient='records')

json_output_path = 'combined_resep.json'
with open(json_output_path, 'w') as json_file:
    json.dump(selected_dict, json_file, indent=4)

print(f"File JSON berhasil disimpan di {json_output_path}")
