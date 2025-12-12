import pandas as pd
import ast
import re 
import os
import google.generativeai as genai
from dotenv import load_dotenv

# ==========================================
# 1. THE DUMMY DATABASE
# ==========================================
MOCK_DATABASE = [
    {
        "Title": "Miso-Butter Roast Chicken with Roasted Radishes",
        "Cleaned_Ingredients": "['chicken', 'miso', 'butter', 'radish', 'pepper']",
        "Instructions": "1. Preheat oven to 400F. 2. Mix miso and butter. 3. Rub on chicken.",
        "Image_Name": "miso-butter-roast-chicken-with-roasted-radishes"
    },
    {
        "Title": "Crispy Salt and Pepper Potatoes",
        "Cleaned_Ingredients": "['potato', 'oil', 'salt', 'pepper', 'rosemary']",
        "Instructions": "1. Cut potatoes. 2. Toss in oil. 3. Roast until crispy.",
        "Image_Name": "crispy-salt-and-pepper-potatoes-drizzle"
    },
    {
        "Title": "Simple Scrambled Eggs",
        "Cleaned_Ingredients": "['egg', 'milk', 'butter', 'salt']",
        "Instructions": "1. Whisk eggs. 2. Melt butter. 3. Cook gently.",
        "Image_Name": "simple-scrambled-eggs"
    }
]

# 1. Load the secret .env file
load_dotenv()

# 2. Get the key safely
api_key = os.getenv("GENAI_API_KEY")

if not api_key:
    raise ValueError("❌ API Key not found! Make sure you created a .env file.")

genai.configure(api_key=api_key)

# ==========================================
# 2. DATA LOADING LOGIC
# ==========================================
def load_data():
    # 1. Get the folder where THIS script (recommender.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Build the path: Current Folder -> FoodDataset -> food_recipes.csv
    csv_path = os.path.join(current_dir, "FoodDataset", "food_recipes.csv")
    
    print(f"📂 Looking for dataset at: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print("✅ Real dataset found! Using full database.")
    except FileNotFoundError:
        print("⚠️ CSV not found. Switching to DUMMY DATA mode.")
        print(f"   (Please ensure 'food_recipes.csv' is inside the 'FoodDataset' folder)")
        df = pd.DataFrame(MOCK_DATABASE)

    # Standard cleaning logic...
    def clean_ingredient_column(x):
        try:
            return ast.literal_eval(x)
        except:
            return x

    df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].apply(clean_ingredient_column)
    return df

# ==========================================
# 3. RECIPE MATCHING ENGINE
# ==========================================
def find_matches(detected_list, df):
    print(f"\n🔎 Matching recipes for: {detected_list}...")
    user_ingredients = set(item.lower() for item in detected_list)
    results = []

    for index, row in df.iterrows():
        if isinstance(row['Cleaned_Ingredients'], list):
            recipe_ing_list = row['Cleaned_Ingredients']
        else:
            continue

        recipe_ingredients = set(item.lower() for item in recipe_ing_list)
        intersection = user_ingredients.intersection(recipe_ingredients)
        
        if len(recipe_ingredients) == 0: continue
            
        match_score = len(intersection) / len(recipe_ingredients)
        
        if len(intersection) >= 1: 
            results.append({
                "Recipe": row['Title'],
                "Match Score": match_score, 
                "You Have": list(intersection),
                "Missing": list(recipe_ingredients - user_ingredients),
                "Image File": str(row['Image_Name']) + ".jpg"
            })
            
    return sorted(results, key=lambda x: x['Match Score'], reverse=True)[:5]

# ==========================================
# 3. GEN-AI CHEF (THE NEW PART)
# ==========================================
def generate_ai_recipe(ingredients):
    print("\n🤖 Database matches were weak. Asking AI Chef for a custom recipe...")
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = (
        f"I have these ingredients in my fridge: {', '.join(ingredients)}. "
        "Create a simple, creative recipe I can cook right now. "
        "Assume I have basic pantry items like oil, salt, and pepper. "
        "Format the output clearly with Title, Ingredients, and Instructions."
    )
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI Error: {e}"

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    df = load_data() 
    
    # TEST CASE: Use random ingredients to force the AI to trigger
    my_ingredients = ["rice", "bread", "chicken"] 
    
    matches = find_matches(my_ingredients, df)
    
    print("\n========= RESULTS =========")
    
    # 1. TRY DATABASE MATCHES FIRST
    high_quality_match_found = False
    
    if matches:
        # Check if the top match is good (e.g., > 50%)
        if matches[0]['Match Score'] > 0.5:
            high_quality_match_found = True
            print("✅ Found great recipes in the database!\n")
            
        for m in matches:
            # Only print if it's a decent match OR if we found nothing else
            if m['Match Score'] > 0.2: 
                print(f"🍲 Recipe: {m['Recipe']}")
                print(f"   Match: {int(m['Match Score']*100)}%")
                print(f"   ⚠️ Missing Ingredients:")
                
                # Smart Cleaning Logic
                for item in m['Missing']:
                    text = str(item).lower().strip()
                    text = re.sub(r'\([^)]*\)', '', text)
                    text = re.sub(r'[\d¼½¾⅛]+', '', text)
                    text = text.replace('/', '').replace('-', '').replace('.', '')
                    words = text.split()
                    forbidden_words = {
                        'g', 'ml', 'oz', 'lb', 'kg', 'tsp', 'tbsp', 'cup', 'cups', 
                        'liter', 'teaspoon', 'tablespoon', 'ounce', 'gram', 'pound',
                        'large', 'small', 'medium', 'fresh', 'dried', 'chopped', 
                        'optional', 'if', 'necessary', 'to', 'serve', 'taste', 'and'
                    }
                    filtered_words = [w for w in words if w not in forbidden_words]
                    text = " ".join(filtered_words).title().strip()
                    
                    if len(text) > 2:
                        print(f"     • {text}")
                print("-" * 30)

    # 2. IF MATCHES ARE BAD (OR NON-EXISTENT), TRIGGER AI
    if not high_quality_match_found:
        print("\n⚠️ No good database matches found.")
        
        # Call the GenAI Function
        ai_recipe = generate_ai_recipe(my_ingredients)
        
        print("\n" + "="*40)
        print("✨ GENERATIVE AI CHEF SUGGESTION ✨")
        print("="*40)
        print(ai_recipe)