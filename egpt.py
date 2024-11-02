import warnings
warnings.filterwarnings("ignore")
import torchvision
torchvision.disable_beta_transforms_warning()


import openai
import pandas as pd
from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class egpt:
    def __init__(self, apiKey, modelName='gpt-4-turbo', embeddingModel='all-MiniLM-L6-v2', datasetPath='hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json'):
        openai.api_key = apiKey
        self.modelName = modelName
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embeddingModel = SentenceTransformer(embeddingModel)
        self.dataset = self.loadDataset(datasetPath)
        self.knowledgeBase = self.createKnowledgeBase()
    
    def loadDataset(self, path):
        dataset = pd.read_json(path, lines=True)
        return dataset[['Context', 'Response']].values.tolist()
    
    def createKnowledgeBase(self):
        knowledgeBase = []
        for context, response in self.dataset:
            embedding = self.embeddingModel.encode(context)
            knowledgeBase.append((embedding, response))
        return knowledgeBase
    
    def getSimilarResponse(self, userContext):
        userEmbedding = self.embeddingModel.encode(userContext)
        similarities = [cosine_similarity([userEmbedding], [kbEmbedding])[0][0] for kbEmbedding, _ in self.knowledgeBase]
        bestMatchIdx = similarities.index(max(similarities))
        _, bestResponse = self.knowledgeBase[bestMatchIdx]
        return bestResponse
    
    def queryGpt(self, context):
        response = openai.ChatCompletion.create(
            model=self.modelName,
            messages=[{'role': 'user', 'content': context}]
        )
        return response.choices[0].message['content']
    
    def respond(self, userContext):
        similarResponse = self.getSimilarResponse(userContext)
        prompt = f'Given the following context and a similar response, please respond appropriately:\n\nContext: {userContext}\n\nSimilar Response: {similarResponse}'
        return self.queryGpt(prompt)
