import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning)
import torchvision
torchvision.disable_beta_transforms_warning()


import os
import re
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import numpy as np
from alive_progress import alive_bar


class Preprocessor:
    def __init__(self, modelName='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(modelName)
        self.labelMap = {
            0: 'Anxiety',
            1: 'Depression',
            2: 'Stress',
            3: 'Happiness',
            4: 'Relationship Issues',
            5: 'Self-Harm',
            6: 'Substance Abuse',
            7: 'Trauma',
            8: 'Obsessive Compulsive Disorder',
            9: 'Eating Disorders',
            10: 'Grief',
            11: 'Phobias',
            12: 'Bipolar Disorder',
            13: 'Post-Traumatic Stress Disorder',
            14: 'Mental Fatigue',
            15: 'Mood Swings',
            16: 'Anger Management',
            17: 'Social Isolation',
            18: 'Perfectionism',
            19: 'Low Self-Esteem',
            20: 'Family Issues'
        }
        
        self.keywords = {
            'anxiety': 0,
            'depressed': 1,
            'sad': 1,
            'stress': 2,
            'happy': 3,
            'relationship': 4,
            'self-harm': 5,
            'substance': 6,
            'trauma': 7,
            'ocd': 8,
            'eating': 9,
            'grief': 10,
            'phobia': 11,
            'bipolar': 12,
            'ptsd': 13,
            'fatigue': 14,
            'mood': 15,
            'anger': 16,
            'isolated': 17,
            'perfectionism': 18,
            'self-esteem': 19,
            'family': 20
        }
    
    def tokenizeText(self, text, maxLength=128):
        return self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=maxLength, 
            return_tensors='pt'
        )
    
    def preprocessDataset(self, texts):
        inputIds, attentionMasks = [], []
        for text in texts:
            encodedDict = self.tokenizeText(text)
            inputIds.append(encodedDict['input_ids'])
            attentionMasks.append(encodedDict['attention_mask'])
        return torch.cat(inputIds, dim=0), torch.cat(attentionMasks, dim=0)

    def labelContext(self, context):
        context = context.lower()
        pattern = r'\b(?:' + '|'.join(re.escape(keyword) for keyword in self.keywords.keys()) + r')\b'
        match = re.search(pattern, context)
        return self.keywords[match.group(0)] if match else None


class etal(Preprocessor):
    def __init__(self, modelName='bert-base-uncased', numLabels=21):
        super().__init__(modelName)
        self.model = BertForSequenceClassification.from_pretrained(modelName, num_labels=numLabels)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, texts, labels, epochs=3, batchSize=8, learningRate=2e-5):
        inputIds, attentionMasks = self.preprocessDataset(texts)
        labels = torch.tensor(labels, dtype=torch.long)

        trainIdx, valIdx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=42)
        trainIds, valIds = inputIds[trainIdx], inputIds[valIdx]
        trainMasks, valMasks = attentionMasks[trainIdx], attentionMasks[valIdx]
        trainLabels, valLabels = labels[trainIdx], labels[valIdx]

        trainData = torch.utils.data.TensorDataset(trainIds, trainMasks, trainLabels)
        valData = torch.utils.data.TensorDataset(valIds, valMasks, valLabels)
        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batchSize, shuffle=True)
        valLoader = torch.utils.data.DataLoader(valData, batch_size=batchSize)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learningRate)
        bestValLoss = float('inf')

        with alive_bar(epochs, title='Training Progress') as bar:
            for epoch in range(epochs):
                totalLoss = 0
                self.model.train()
                for i, batch in enumerate(trainLoader):
                    batchIds, batchMasks, batchLabels = batch
                    self.model.zero_grad()

                    outputs = self.model(input_ids=batchIds, attention_mask=batchMasks, labels=batchLabels)
                    loss = outputs.loss
                    totalLoss += loss.item()
                    loss.backward()
                    optimizer.step()

                    print(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(trainLoader)}, Loss: {loss.item()}")

                avgTrainLoss = totalLoss / len(trainLoader)
                valLoss = self.evaluate(valLoader)
                if valLoss < bestValLoss:
                    bestValLoss = valLoss
                    self.save('models', f'e{epoch}l{valLoss}.pt')
                    print(f"Model State Dict Saved at: {os.path.join(os.getcwd(), 'models', f'e{epoch}l{valLoss}.pt')}")
                print(f'Epoch {epoch + 1}, Train Loss: {avgTrainLoss}, Validation Loss: {valLoss}')
                bar()

    def evaluate(self, dataLoader):
        self.model.eval()
        predictions, trueLabels = [], []
        totalLoss = 0
        with torch.no_grad():
            for batch in dataLoader:
                batchIds, batchMasks, batchLabels = batch
                outputs = self.model(input_ids=batchIds, attention_mask=batchMasks, labels=batchLabels)
                logits = outputs.logits
                loss = outputs.loss
                totalLoss += loss.item()
                predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
                trueLabels.extend(batchLabels.cpu().numpy())
        print(classification_report(trueLabels, predictions))
        return totalLoss / len(dataLoader)

    def predict(self, text):
        self.model.eval()
        tokens = self.tokenizeText(text)
        with torch.no_grad():
            outputs = self.model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
            prediction = torch.argmax(outputs.logits, axis=1).item()
        return self.labelMap.get(prediction)

    def save(self, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        torch.save(self.model.state_dict(), filepath)

    def load(self, filePath, best = True):
        if best:
            modelFiles = [f for f in os.listdir(filePath) if f.endswith('.pt')]
            if not modelFiles:
                print('No model files found in the specified folder.')
                return

            modelFiles.sort(key=lambda x: (int(x.split('e')[1].split('l')[0]), float(x.split('l')[1].split('.')[0])))
            
            bestModelFile = modelFiles[-1]
            modelPath = os.path.join(filePath, bestModelFile)
            self.model.load_state_dict(torch.load(modelPath))
        else:
            self.model.load_state_dict(torch.load(filePath))

        print(f'Loaded model state dict')
        self.model.eval()