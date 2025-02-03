from flask import Flask, request, jsonify
import torch
from training import MyMobileBert
from transformers import MobileBertTokenizer


model = MyMobileBert.from_pretrained("google/mobilebert-uncased", num_labels=2)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")

app = Flask(__name__)

@app.route('/predict', methods=["POST"])
def predict():
    data = request.json
    text = data.get("text")

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    with torch.no_grad():
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs
        prediction = torch.argmax(logits, dim=1).item()

    return jsonify({'prediction': prediction})