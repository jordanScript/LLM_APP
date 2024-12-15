from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Importamos el tokenizador
model_name = "google/flan-t5-small"

tokenizer_FT5 = T5Tokenizer.from_pretrained(model_name, legacy=True)
# Importamos el modelo pre-entrenado
model_FT5 = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

@app.route('/flan-t5', methods=['POST'])
def query():
    data = request.json
    prompt = f"""{data.get('prompt')}"""

    prompt_template = f"Resume el siguiente articulo:\n\n{prompt}"


    # Tokenizamos el prompt
    prompt_tokens = tokenizer_FT5(prompt_template, return_tensors="pt").input_ids.to("cuda")

    # Generamos los siguientes tokens
    outputs = model_FT5.generate(prompt_tokens, max_length=200)
    
    return tokenizer_FT5.decode(outputs[0],skip_special_tokens=True)
if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=5000, debug=True) 
    app.run(port=5000, debug=True)
    # pass