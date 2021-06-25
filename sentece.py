import numpy as np
from pyknp import Juman
from sentence_transformers import SentenceTransformer
import scipy.spatial

if __name__ == "__main__":

    model_path = "/home/yamada/Downloads/training_bert_japanese"
    model = SentenceTransformer(model_path, show_progress_bar=False)
    sentences = ["お辞儀をしている男性会社員", "笑い袋", "テクニカルエバンジェリスト（女性）", "戦うAI", "笑う男性（5段階）",
        "お金を見つめてニヤけている男性", "「ありがとう」と言っている人", "定年（女性）", "テクニカルエバンジェリスト（男性）", "スタンディングオベーション"]
    sentence_vectors = model.encode(sentences)
    
    queries = ['暴走したAI', '暴走した人工知能', 'いらすとやさんに感謝', 'つづく']
    query_embeddings = model.encode(queries)

    closest_n = 5
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_vectors, metric="cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for idx, distance in results[0:closest_n]:
            print(sentences[idx].strip(), "(Score: %.4f)" % (distance / 2))