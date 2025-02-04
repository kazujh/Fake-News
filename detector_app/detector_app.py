from flask import Flask, request, jsonify
import torch
import os
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification

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

model = MyMobileBert.from_pretrained("google/mobilebert-uncased", num_labels=2)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")

def create_app(test_config=None):

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite')
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)


    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

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
    return app