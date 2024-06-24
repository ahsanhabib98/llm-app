import re
import os
import torch

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)

# Path to your local model directory
local_model_path = '/home/notebooks/lama-2-7B-agenda'
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(local_model_path, local_files_only=True)


# Define endpoint for prediction
@app.route('/api/predict-opinion', methods=['POST'])
@cross_origin()
def predict_opinion():
    if request.is_json:
        data = request.get_json()
        user_prompt = data.get('user_prompt')
        assemply_member = data.get('assemply_member')
        prompt = f"Suppose you are {assemply_member}. Given the agenda '{user_prompt}', breifly mention three strategy to tackle it"

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, device_map="auto")
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        generated_text = result[0]['generated_text']
        content_after_inst = generated_text.split("[/INST]", 1)[-1]
        response = re.sub(r'\</s\>', '', content_after_inst).strip()

        lines = response.split('\n')
        html_list_items = []
        for line in lines:
            if line:
                html_list_items.append(f"<li>{line}</li>")
        html_output = ''.join(html_list_items)


    
        # Return the predicted data
        return jsonify({'html_output': html_output, 'response': response}), 200
    else:
        return jsonify({'error': 'Request must be in JSON format'}), 400


# Define endpoint for prediction
@app.route('/api/predict-outcome', methods=['POST'])
@cross_origin()
def predict_outcome():
    if request.is_json:
        data = request.get_json()
        user_prompt = data.get('user_prompt')
        prompt = f"Given the assembply members' responses '{user_prompt}', breifly summarize the response in three numbered response"

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, device_map="auto")
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        generated_text = result[0]['generated_text']
        content_after_inst = generated_text.split("[/INST]", 1)[-1]
        response = re.sub(r'\</s\>', '', content_after_inst).strip()

        lines = response.split('\n')
        html_list_items = []
        for line in lines:
            if line:
                html_list_items.append(f"<li>{line}</li>")
        html_output = ''.join(html_list_items)
    
        # Return the predicted data
        return jsonify({'html_output': html_output, 'response': response}), 200
    else:
        return jsonify({'error': 'Request must be in JSON format'}), 400
    

# Define endpoint for prediction
@app.route('/api/predict-next-opinion', methods=['POST'])
@cross_origin()
def predict_next_opinion():
    if request.is_json:
        data = request.get_json()
        user_prompt = data.get('user_prompt')
        prompt = f"Given this summary '{user_prompt}', breifly suggest five new agenda having at max 10 words each, in three numbered response."

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, device_map="auto")
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        generated_text = result[0]['generated_text']
        content_after_inst = generated_text.split("[/INST]", 1)[-1]
        response = re.sub(r'\</s\>', '', content_after_inst).strip()

        lines = response.split('\n')
        html_list_items = []
        for line in lines:
            if line:
                html_list_items.append(f"<li>{line}</li>")
        html_output = ''.join(html_list_items)
    
        # Return the predicted data
        return jsonify({'html_output': html_output, 'response': response}), 200
    else:
        return jsonify({'error': 'Request must be in JSON format'}), 400


if __name__ == '__main__':
    app.run()
