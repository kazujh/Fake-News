# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from transformers import BertTokenizer

# %%
#data preprocessing
SEED = 42

# %%
file_directory = os.getcwd()

# %%
def preprocess(text):
    text = text.lower()
    if text[:4] == 'says':
        text = text[4:]
    elif text[:5] == 'print':
        text = text[5:]
    text = re.sub(r'-+', ' ', text)
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'(\n)+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

# %%
def combine_dfs(df1, df2):
    return pd.concat([df1, df2])

# %%
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
import random

# %%
def paraphrase(text):
    words = text.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            add = random.choice(synonyms[0].lemmas())
            new_words.append(add.name())
        else:
            new_words.append(word)
    return ' '.join(new_words)

# %%
column_names = ['id', 'truth-value', 'statement', 'topics', 'speaker', 'speaker occupation', 'state', 'party', 'barely-true', 'false', 'half-true', 'true', 'POF', 'context']
df = pd.DataFrame()
current_df = pd.read_csv("./Liar dataset/train.tsv", sep="\t", names=column_names)


label_map = {
    'POF': 0,
    'false': 0,
    'barely-true': 0,
    'half-true': 1,
    'true': 1
}
label_columns = ['POF', 'false', 'barely-true', 'half-true', 'true']
current_df = current_df.dropna(subset=label_columns, how='all')
current_df = current_df[['statement', 'POF', 'false', 'barely-true', 'half-true', 'true']]
current_df['label'] = current_df[label_columns].idxmax(axis=1)
current_df['truth'] = current_df['label'].map(label_map)

"""weighted_sum = sum([current_data[col] * label_map[col] for col in label_columns])
total_counts = current_data[label_columns].sum(axis=1)
current_data['confidence'] = round((weighted_sum / total_counts), 2)"""
current_df = current_df[['statement', 'truth']]
current_df['statement'] = current_df['statement'].apply(preprocess)

# %%
current_df = current_df.sample(frac=1).reset_index(drop=True)

# %%
paraphrased_statements = pd.Series([paraphrase(statement) for statement in current_df[:len(current_df) // 2]['statement']])

# %%
paraphrased_statements = paraphrased_statements.apply(preprocess)
paraphrased_df = pd.DataFrame({'statement': paraphrased_statements, 'truth': current_df['truth'][:len(current_df) // 2]})

# %%
current_df = combine_dfs(paraphrased_df, current_df)

# %%
oversampler = RandomOverSampler(random_state=SEED)
X_resampled, y_resampled = oversampler.fit_resample(current_df[['statement']], current_df['truth'])
oversampled_df = pd.DataFrame({'statement': X_resampled['statement'], 'truth': y_resampled})

# %%
df = combine_dfs(df, oversampled_df)

# %%
current_df = pd.read_csv('./dataset 1/FakeNewsNet.csv')

current_df = current_df[['title', 'real']]
current_df = current_df.rename(columns={'title':'statement', 'real':'truth'})
current_df['truth'] = current_df['truth'].astype(int)
current_df['statement'] = current_df['statement'].apply(preprocess)

# %%
current_df = current_df.sample(frac=1).reset_index(drop=True)

# %%
paraphrased_statements = paraphrased_statements.apply(preprocess)
paraphrased_df = pd.DataFrame({'statement': paraphrased_statements, 'truth': current_df['truth'][:len(current_df) // 2]})

# %%
current_df = combine_dfs(paraphrased_df, current_df)

# %%
X_resampled, y_resampled = oversampler.fit_resample(current_df[['statement']], current_df['truth'])
oversampled_df = pd.DataFrame({'statement': X_resampled['statement'], 'truth': y_resampled})

# %%
df = combine_dfs(df, oversampled_df)

# %%
current_df = pd.read_csv('./KaggleFakeNews/train.csv')

# %%
current_df = current_df.dropna(how='all')
current_df = current_df.rename(columns={'text':'statement', 'label':'truth'})
current_df = current_df[['statement', 'truth']]
current_df['statement'] = current_df['statement'].astype(str).apply(preprocess)

# %%
current_df = current_df.sample(frac=1).reset_index(drop=True)

# %%
paraphrased_statements = pd.Series([paraphrase(statement) for statement in current_df['statement']])
paraphrased_df = pd.DataFrame({'statement': paraphrased_statements, 'truth': current_df['truth'][:len(current_df) // 2]})

# %%
current_df = combine_dfs(paraphrased_df, current_df)

# %%
current_df = current_df.dropna()

# %%
X_resampled, y_resampled = oversampler.fit_resample(current_df[['statement']], current_df['truth'])
oversampled_df = pd.DataFrame({'statement': X_resampled['statement'], 'truth': y_resampled})

# %%
df = combine_dfs(df, oversampled_df)

# %%
df['truth'].value_counts()

# %%
from scipy.sparse import issparse, csr_matrix
from sklearn.preprocessing import MinMaxScaler

# %%
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self.grow_tree(X, y, 0)

    def grow_tree(self, X, y, depth):

        if depth >= self.max_depth or X.shape[0] < self.min_samples_split:
            return np.bincount(y).argmax()

        best_feature, best_threshold = self.find_best_split(X, y)
        if best_feature is None:
            return np.bincount(y).argmax()
        X_col = X[:, best_feature].toarray().flatten()    
        left_indices = X_col <= best_threshold
        right_indices = X_col > best_threshold

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            # Return majority class if split is invalid
            return np.bincount(y).argmax()
        left = self.grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self.grow_tree(X[right_indices], y[right_indices], depth + 1)
        return {'feature': best_feature, 'threshold': best_threshold, 'left': left, 'right': right}

    def find_best_split(self, X, y):
        #iterate through every split and test gini
        n_features = X.shape[1]
        features = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        best_gini = 1.0
        best_feature, best_threshold = None, None
        for feature in features:
            X_col = X[:, feature].toarray().flatten()
            thresholds = np.unique(X_col[X_col > 0])
            for threshold in thresholds:
                left_indices = X_col <= threshold
                right_indices = X_col > threshold
                groups = [y[left_indices], y[right_indices]]

                gini = gini_impurity(groups, np.unique(y))

                if gini < best_gini:
                    best_gini = gini
                    best_threshold = threshold
                    best_feature = feature
            
        return best_feature, best_threshold

    def _predict_tree(self, X):
        predictions = []
        for row in X:
            node = self.tree
            while isinstance(node, dict):
                if row[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(node)
        return predictions

# %%
def gini_impurity(groups, classes):
    n_instances = sum([len(group) for group in groups])
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = list(group).count(class_val) / size
            score += proportion ** 2
        gini += (1.0 - score) * (size / n_instances)
    return gini

# %%
class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            X_sample, y_sample = random_sample(X, y)

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        X_dense = X.toarray() if hasattr(X, "toarray") else X  # Handle sparse matrices
        predictions = np.array([tree._predict_tree(X_dense) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

# %%
def random_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]

# %%
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# %%
#model = RandomForestClassifier(n_estimators=100, random_state=42)
#model.fit(X_train, y_train)

# %%
#y_pred = model.predict(X_test)

# %%
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))

# %%
#assume X parameter will be sparse
#print(X_train[0].indices)
#print(X_train[0].data)

# %%
#rf = RandomForest(n_trees=100, max_depth=10000)
#rf.fit(X_train, y_train)

#y_pred = rf.predict(X_test)

#accuracy = np.sum(y_pred == y_test) / len(y_test)
#print(f'accuracy: {accuracy}')

# %%
import tensorflow as tf
from transformers import MobileBertTokenizer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification

# %%
from fastai.text.all import *

# %%
df = df.sample(frac=1).reset_index(drop=True)

# %%
"""X_bert = df['statement'].astype(str)
y_bert = np.array(df['truth'])
X_bert_train, X_bert_test, y_bert_train, y_bert_test = train_test_split(X_bert, y_bert, test_size=.2, random_state=42)"""

# %%
#tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

# %%
#X_bert_train = X_bert_train.tolist()
#X_bert_test = X_bert_test.tolist()

# %%
"""train_tokenized = tokenizer(
    X_bert_train,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors='pt'
)

test_tokenized = tokenizer(
    X_bert_test,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors='pt'
)"""

# %%
import torch
from torch.utils.data import Dataset, DataLoader

"""class MyBertDataset(Dataset):
    def __init__(self, encodings, labels):
        """
        encodings: a dict of Tensors (e.g., 'input_ids', 'attention_mask') from tokenizer(..., return_tensors='pt')
        labels: a list or tensor of labels (e.g., y_bert_train)
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        # Number of examples
        return len(self.labels)

    def __getitem__(self, idx):
        # For the given idx, gather all the tokenized inputs
        input_ids = self.encodings['input_ids'][idx].clone().detach()
        attention_mask = self.encodings['attention_mask'][idx].clone().detach()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, label
    def __testgetitem__(self, idx):
        # Suppose we load data from somewhere and convert it to tensors
        input_ids = torch.tensor([...])
        attention_mask = torch.tensor([...])
        label = torch.tensor(...)
        return input_ids, attention_mask, label"""

# %%
#train_ds = MyBertDataset(train_tokenized, y_bert_train)
#test_ds  = MyBertDataset(test_tokenized,  y_bert_test)

# %%
from fastai.vision.all import *

# %%
"""def collate(batch):
    # Suppose each batch item is a tuple: (input_ids_list, attention_mask_list, label)
    input_ids = []
    attention_masks = []
    labels = []
    
    for b in batch:
        input_ids.append(b[0])
        attention_masks.append(b[1])
        labels.append(b[2])
        
    # Convert them all to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return (input_ids, attention_masks, labels)
"""

# %%
"""train_dl = DataLoader(train_ds, batch_size=2, collate_fn=collate)
test_dl = DataLoader(test_ds, batch_size=2, collate_fn=collate)

dls = DataLoaders(train_dl, test_dl)"""

# %%
df = df.reset_index()

# %%
print(type(df))

# %%
dblock = DataBlock(
    blocks=(TextBlock.from_df('statement', seq_len=128), CategoryBlock),  # Text & Labels
    get_x=ColReader('statement'),  # Feature column
    get_y=ColReader('truth'),  # Label column
    splitter=RandomSplitter(valid_pct=0.2)  # Split into train/validation sets
)
print(dblock.datasets(df).items)
# Load data into DataLoader
dls = dblock.dataloaders(df, bs=16)

# %%
learn = text_classifier_learner(dls, AWD_LSTM, metrics=[accuracy])

# %%
from fastai.callback.transformers import HF_TextBlock, HF_Trainer

# Define Hugging Face tokenizer and model
hf_model = 'bert-base-uncased'
dblock = DataBlock(
    blocks=(HF_TextBlock(hf_model), CategoryBlock),
    get_x=ColReader('text'),
    get_y=ColReader('label'),
    splitter=RandomSplitter(valid_pct=0.2)
)

dls = dblock.dataloaders(data, bs=16)
learn = HF_Trainer(dls, model=AutoModelForSequenceClassification.from_pretrained(hf_model, num_labels=2), metrics=[accuracy])

# %%
learn.fit_one_cycle(3, .0002)

# %%
learn.validate()

# %%
# Get one batch from your train_dataloader
"""batch = next(iter(train_dl))

# If your batch is something like (input_ids, attention_mask, labels):
input_ids, attention_mask, labels = batch

# Now check their types:
print(type(input_ids), type(attention_mask), type(labels))"""

# %%
from transformers import MobileBertForSequenceClassification

class MyMobileBert(MobileBertForSequenceClassification):
    def forward(self, input_ids=None, attention_mask=None, labels=None, return_loss=False, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        if return_loss and labels is not None:
            return outputs.logits, outputs.loss
        # Return only the logits, which is what the fastai Learner will treat as predictions
        return outputs.logits


# Then instantiate:
model = MyMobileBert.from_pretrained("google/mobilebert-uncased", num_labels=2)

# %%
from fastai.optimizer import Adam

# %%
learn = Learner(
    dls,
    model,
    loss_func=CrossEntropyLossFlat(),
    metrics=accuracy,
)

# 9) Use lr_find()
lr = learn.lr_find()
print("lr_find results:", lr)

# %%
from transformers import AdamW

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=lr.valley)

# %%
#lr.valley = .0002

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
for epoch in range(5):
    model.train()
    for batch in train_dl:
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_loss=True)
        loss = outputs[1]
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dl:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            # Forward pass with validation loss computation
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_loss=True)
            logits = outputs[0]  # Logits from the model
            val_loss = F.cross_entropy(logits, labels)  # Compute loss for this batch
            
            # Accumulate total loss
            total_val_loss += val_loss.item()
            
            # Calculate predictions and accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Calculate average validation loss and accuracy
    avg_val_loss = total_val_loss / len(test_dl)  # Average loss per batch
    accuracy = correct / total  # Overall accuracy

    # Print metrics for this epoch
    print(f"Epoch {epoch + 1}:")
    print(f"  Average Validation Loss: {avg_val_loss:.4f}")
    print(f"  Validation Accuracy: {accuracy * 100:.2f}%")

# %%
print('done')

# %%
torch.save(model.state_dict(), "my_model_weights.pt")

# %%
# Recreate the same model architecture
model = MyMobileBert.from_pretrained("google/mobilebert-uncased", num_labels=2)  # same class & config used before

# Load state dict
model.load_state_dict(torch.load("my_model_weights.pt"))
model.eval()  # or model.train() if you're continuing training

# %%
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# %%
from tensorflow.keras.mixed_precision import Policy, set_global_policy

# Set the global policy for mixed precision
policy = Policy('mixed_float16')
set_global_policy(policy)

print("Mixed precision policy set:", policy)

# %%
history = model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# %%
model.save_pretrained('./new_model')
tokenizer.save_pretrained('./new_model')

# %%
test_data = pd.read_csv("./Liar dataset/test.tsv", sep="\t", names=column_names)

label_map = {
    'POF': 0,
    'false': 0,
    'barely-true': 0,
    'half-true': 1,
    'true': 1
}
label_columns = ['POF', 'false', 'barely-true', 'half-true', 'true']

test_data = test_data.dropna(subset=label_columns, how='all')
test_data = test_data[['statement', 'POF', 'false', 'barely-true', 'half-true', 'true']]
test_data['label'] = test_data[label_columns].idxmax(axis=1)
test_data['truth'] = test_data['label'].map(label_map)

"""weighted_sum = sum([current_data[col] * label_map[col] for col in label_columns])
total_counts = current_data[label_columns].sum(axis=1)
current_data['confidence'] = round((weighted_sum / total_counts), 2)"""
test_data = test_data[['statement', 'truth']]
test_data['statement'] = test_data['statement'].apply(preprocess)

# %%
feats = test_data['statement'].astype(str).tolist()
labels = test_data['truth']

tst_tokenized = tokenizer(
    feats,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors='tf'
)
tst_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': tst_tokenized['input_ids'], 'attention_mask': tst_tokenized['attention_mask']},
    labels
)).batch(16)

# %%


# %%
from tqdm import tqdm

batch_size = len(test_tokenized['input_ids']) // 32  # Adjust based on your hardware
outputs = []

for i in tqdm(range(0, len(test_tokenized['input_ids']), batch_size)):
    batch_input_ids = test_tokenized['input_ids'][i:i+batch_size]
    batch_attention_mask = test_tokenized['attention_mask'][i:i+batch_size]
    batch_output = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
    outputs.append(batch_output.logits)

# %%
all_logits = tf.concat(outputs, axis=0)

print(all_logits.shape)

# %%
predicted_classes = tf.argmax(all_logits, axis=-1)
correct_predictions = tf.reduce_sum(tf.cast(predicted_classes == y_bert_test, tf.float32))
accuracy = correct_predictions / len(y_bert_test)

print(f"Model Accuracy: {accuracy.numpy():.2%}")

# %%


# %%
"""current_data_1 = pd.read_csv('./dataset 2/dataset/gossipcop_fake.csv')
current_data_1 = current_data_1.dropna(how='all')
current_data_1 = current_data_1[['title']]
current_data_1 = current_data_1.rename(columns={'title':'statement'})
current_data_1['truth'] = 0"""

# %%
"""current_data_2 = pd.read_csv('./dataset 2/dataset/gossipcop_real.csv')
current_data_2 = current_data_2.dropna(how='all')
current_data_2 = current_data_2[['title']]
current_data_2 = current_data_2.rename(columns={'title':'statement'})
current_data_2['truth'] = 1"""

# %%
"""current_data = pd.concat([current_data_1, current_data_2])
X = current_data['statement']
y = current_data['truth']

vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_vectorized, y)

resampled_df = pd.DataFrame({'statement': vectorizer.inverse_transform(X_resampled),  # Attempt to reverse transform
                             'truth': y_resampled})
current_data = resampled_df"""

# %%
#data = pd.concat([data, current_data])

# %%
"""current_df_1 = pd.read_csv('./dataset 2/dataset/politifact_fake.csv')
current_df_1 = current_df_1.dropna(how='all')
current_df_1 = current_df_1[['title']]
current_df_1 = current_df_1.rename(columns={'title':'statement'})
current_df_1['truth'] = 0
current_df_1['statement'] = current_df_1['statement'].apply(preprocess)"""

# %%
"""current_df_2 = pd.read_csv('./dataset 2/dataset/politifact_real.csv')
current_df_2 = current_df_2.dropna(how='all')
current_df_2 = current_df_2[['title']]
current_df_2 = current_df_2.rename(columns={'title':'statement'})
current_df_2['truth'] = 1
current_df_2['statement'] = current_df_2['statement'].apply(preprocess)"""

# %%
#current_df = pd.concat([current_df_1, current_df_2])
"""X = current_data['statement']
y = current_data['truth']

vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_vectorized, y)

resampled_df = pd.DataFrame({'statement': vectorizer.inverse_transform(X_resampled),  # Attempt to reverse transform
                             'truth': y_resampled})
current_data = resampled_df"""

# %%
#logistic regression model from scratch

# %%
from sklearn.metrics import accuracy_score, classification_report

# %%
class LogisticRegression:
    def __init__(self):
        ...
        
    def sigmoid(self, n):
        return 1 / (1 + np.exp(-n))
        
    def initialize_weights(self, n_features):
        weights = np.zeros(n_features)
        bias = 0
        return weights, bias
        
    def predict(self, X, weights, bias):
        linear_model = X.dot(weights) + bias
        predictions = self.sigmoid(linear_model)
        return predictions
        
    def calculate_loss(self, y_true, y_pred):
        n = len(y_true)
        loss = (-1/n) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def gradient_descent(self, X, y, weights, bias, lr):
        n = X.shape[0]
    
        y_pred = self.predict(X, weights, bias)
    
        dw = X.T.dot(y_pred - y) / n
        db = np.sum(y_pred - y) / n
    
        weights -= lr * dw
        bias -= lr * db
    
        return weights, bias

    def train(self, X, y, lr=.1, epochs=1000, batch_size=500):
        n_features = X.shape[1]
    
        weights, bias = self.initialize_weights(n_features)
    
        losses = []
    
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                
                
                weights, bias = self.gradient_descent(X_batch, y_batch, weights, bias, lr)
    
            y_pred = self.predict(X, weights, bias)
            loss = self.calculate_loss(y, y_pred)
            losses.append(loss)
    
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        return weights, bias, losses

    def classify(self, X, weights, bias, threshold=.5):
        probabilities = self.predict(X, weights, bias)
        return [1 if p >= threshold else 0 for p in probabilities]

# %%
#USE FOR THE SKLEARN MODEL
#vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

#X = data['statement'].astype(str)

# %%
#USE FOR THE SKLEARN MODEL
"""X_tfidf = vectorizer.fit_transform(X)
y = np.array(data['truth'])"""

# %%
#X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=.2, random_state=42) #USE FOR THE SKLEARN MODEL

# %%
#DON'T USE FOR SKLEARN
#regressor = LogisticRegression()
#weights, bias, losses = regressor.train(X_train, y_train)
#y_pred = regressor.classify(X_test, weights, bias)
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))

# %%
#logistic regression using sklearn

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# %%
#model = LogisticRegression(max_iter=1000, solver='lbfgs')
#model.fit(X_train, y_train)

# %%
#y_pred = model.predict(X_test)
#y_prob = model.predict_proba(X_test)[:, 1]

# %%
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))

# %%
#random forest

# %%



