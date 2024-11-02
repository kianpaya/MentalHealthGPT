from etal import *
import pandas as pd


dataset = pd.read_json('hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json', lines=True)

model = etal()

texts = dataset['Context'].tolist()
labels = dataset['Response'].apply(lambda x: model.labelContext(x)).tolist()
labels = [int(label) if not np.isnan(label) else 15 for label in labels]
labels = torch.tensor(labels, dtype=torch.long)

model.train(texts, labels, epochs=1, batchSize=8, learningRate=2e-5)

model.load('models')

sampleText = 'I\'m feeling very anxious about my upcoming exams.'
predictedLabel = model.predict(sampleText)
print('Predicted Label for etal:', predictedLabel)
