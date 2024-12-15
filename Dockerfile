FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Instala Python y dependencias
RUN apt-get update && apt-get install -y python3 python3-pip && apt-get clean

# Directorio de trabajo
WORKDIR /app

# Instala las dependencias
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Expone el puerto
EXPOSE 5000
