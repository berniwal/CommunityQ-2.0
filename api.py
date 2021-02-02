import flask
from flask import request, jsonify
import torch
import torch.nn as nn
from train import SimpleQuestionAnswerer

PATH = './final_model.pt'

model = torch.load(PATH)
model.eval()

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "<h1></h1><p>This site corresponds to a Digitec Application Project, " \
           "please create GET request with fields 'visits' and 'text_length' to /api/prediction " \
           "to create a prediction.</p>"


@app.route('/api/prediction', methods=['GET'])
def prediction():
    if 'visits' in request.args:
        visits = float(request.args['visits'])
        visits = (visits - model.mean.VisitsLastYear) / model.std.VisitsLastYear
    else:
        return "Error: No visits field provided. Please specify number of visits of product page."

    if 'text_length' in request.args:
        text_length = float(request.args['text_length'])
        text_length = (text_length - model.mean.QuestionTextLength) / model.std.QuestionTextLength
    else:
        return "Error: No text length field provided. Please specify text length of comment."

    x = torch.tensor([visits, text_length], dtype=torch.float32)
    out = model(x)
    out = nn.Softmax(dim=0)(out)
    print(out)

    result = {
        'IsQuestionForCommunity': out[1].item()
    }
    return result

app.run(host='0.0.0.0')
