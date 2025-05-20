import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import os

class SentimentAnalyzer:
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        """Initialise le modèle BERT et le tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def preprocess_text(self, text):
        """Tokenise le texte pour BERT."""
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return inputs

    def predict_sentiment(self, text):
        """Prédit le sentiment d'un texte donné."""
        inputs = self.preprocess_text(text)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

        # Le modèle retourne des scores de 1 (très négatif) à 5 (très positif)
        sentiment_map = {0: "Très négatif", 1: "Négatif", 2: "Neutre", 3: "Positif", 4: "Très positif"}
        return sentiment_map.get(predicted_class, "Neutre"), probs[0].cpu().numpy()

    def analyze_dataset(self, data, text_column="text", max_samples=None):
        """Analyse un dataset de textes et retourne les sentiments."""
        if max_samples:
            data = data.head(max_samples)  # Limiter le nombre de critiques
        results = []
        for text in data[text_column]:
            sentiment, probs = self.predict_sentiment(text)
            results.append({"text": text, "sentiment": sentiment, "probabilities": probs})
        return pd.DataFrame(results)

    def save_results(self, results_df, output_path="results.csv"):
        """Sauvegarde les résultats dans un fichier CSV."""
        results_df.to_csv(output_path, index=False)
        print(f"Résultats sauvegardés dans {output_path}")

    def plot_sentiment_distribution(self, results_df, output_path="visuals/sentiment_distribution.png"):
        """Génère un graphique de répartition des sentiments."""
        plt.figure(figsize=(8, 6))
        sns.countplot(data=results_df, x="sentiment", order=["Très négatif", "Négatif", "Neutre", "Positif", "Très positif"])
        plt.title("Répartition des sentiments dans le dataset")
        plt.xlabel("Sentiment")
        plt.ylabel("Nombre de critiques")
        os.makedirs("visuals", exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"Graphique sauvegardé dans {output_path}")

# Exemple d'utilisation avec le dataset IMDb
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    # Charger le dataset IMDb (limité à 100 échantillons pour la démonstration)
    print("Chargement du dataset IMDb...")
    dataset = load_dataset("imdb", split="train")
    train_data = pd.DataFrame(dataset).head(100) 

    print("Analyse des sentiments...")
    results = analyzer.analyze_dataset(train_data, text_column="text", max_samples=100)

    analyzer.save_results(results, "results.csv")

    analyzer.plot_sentiment_distribution(results)

    print(results[["text", "sentiment"]].head())