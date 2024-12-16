from flask import Flask, request
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, concatenate_datasets
import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Leer una variable del archivo .env
model_name = os.getenv("T5_MODEL")

tokenizer_FT5 = T5Tokenizer.from_pretrained(model_name, legacy=True)
# Importamos el modelo pre-entrenado
model_FT5 = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

# Importamos el tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Cargamos el dataset
ds = load_dataset("mlsum", 'es', trust_remote_code=True)


def dataset_reduction_data(ds):
    # Reducimos el conjunto de datos
    NUM_EJ_TRAIN = 1500
    NUM_EJ_VAL = 500
    NUM_EJ_TEST = 200

    # Subconjunto de entrenamiento
    ds['train'] = ds['train'].select(range(NUM_EJ_TRAIN))

    # Subconjunto de validación
    ds['validation'] = ds['validation'].select(range(NUM_EJ_VAL))

    # Subconjunto de pruebas
    ds['test'] = ds['test'].select(range(NUM_EJ_TEST))

    return ds


def set_dataset(ds):
    """Procesa el dataset para adaptarlo a la plantilla."""
    ds["train"] = ds["train"].map(parse_dataset)
    ds["validation"] = ds["validation"].map(parse_dataset)
    ds["test"] = ds["test"].map(parse_dataset)
    return ds

def parse_dataset(ejemplo):
  """Procesa los ejemplos para adaptarlos a la plantilla."""
  return {"prompt": f"Resume el siguiente articulo:\n\n{ejemplo['text']}"}

def calculate_max_promt_length(ds):
    # Calculamos el tamaño máximo de prompt
    prompts_tokens = concatenate_datasets([ds["train"], ds["validation"], ds["test"]]).map(lambda x: tokenizer(x["prompt"], truncation=True), batched=True) # Va a truncar en 512 que es el tamaño máximo para este modelo
    max_token_len = max([len(x) for x in prompts_tokens["input_ids"]])
    print(f"Maximo tamaño de prompt: {max_token_len}")
    return max_token_len

def calculate_max_completions_length(ds):
    # Calculamos el tamaño máximo de completion
    completions_tokens = concatenate_datasets([ds["train"], ds["validation"], ds["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True)
    max_completion_len = max([len(x) for x in completions_tokens["input_ids"]])
    return max_completion_len


def padding_tokenizer(datos):
  
    max_token_len = calculate_max_promt_length(ds)
    max_completion_len = calculate_max_completions_length(ds)

    # Tokenizar inputs (prompts)
    model_inputs = tokenizer(datos['prompt'], max_length=max_token_len, padding="max_length", truncation=True)

    # Tokenizar labels (completions)
    model_labels = tokenizer(datos['summary'], max_length=max_completion_len, padding="max_length", truncation=True)

    # Sustituimos el caracter de padding de las completion por -100 para que no se tenga en cuenta en el entrenamiento
    model_labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_labels["input_ids"]]

    model_inputs['labels'] = model_labels["input_ids"]

    return model_inputs

def query(prompt):
    prompt_template = f"Resume el siguiente articulo:\n\n{prompt}"

    # Aplicamos reducción de datos
    ds = dataset_reduction_data(ds)

    # Procesamos el dataset
    ds = set_dataset(ds)

    # Tokenizamos los datos
    ds_tokens = ds.map(padding_tokenizer, batched=True, remove_columns=['text', 'summary', 'topic', 'url', 'title', 'date', 'prompt'])

    # Tokenizamos el prompt
    prompt_tokens = tokenizer_FT5(prompt_template, return_tensors="pt").input_ids.to("cuda")

    # Generamos los siguientes tokens
    outputs = model_FT5.generate(prompt_tokens, max_length=200)
    
    return tokenizer_FT5.decode(outputs[0],skip_special_tokens=True)