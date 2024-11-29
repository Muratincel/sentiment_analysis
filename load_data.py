import os
import pandas as pd
from preprocess import preprocess_dataframe

def load_data(directory):
    texts = []
    labels = []
    for label in ['pos','neg']:
        folder = os.path.join(directory, label)
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(1 if label == 'pos' else 0)
    return pd.DataFrame({'text': texts, 'label': labels })

train_data = load_data('aclImdb/train')
test_data = load_data('aclImdb/test')

# df = load_data('aclImdb/train')
# df = preprocess_dataframe(df)
df = preprocess_dataframe(train_data)

# print(train_data.head())
# print(df.head())

df.to_csv('preprocessed_data.csv', index=False)
print("Preprocessed data saved to preprocessed_data.csv.")
