import os
import random
from urllib.parse import urlparse
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator, load_img
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

app = Flask(__name__, template_folder='templates', static_folder='')

dataset_path = 'Animals'

def get_train_generator():
    train_path = os.path.join(dataset_path, 'train')
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical'
    )
    return train_generator

def classify_random_image():
    train_generator = get_train_generator()

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_generator, epochs=15)

    random_class = random.choice(list(train_generator.class_indices.keys()))
    random_image_path = os.path.join(dataset_path, 'train', random_class, random.choice(os.listdir(os.path.join(dataset_path, 'train', random_class))))
    random_image = load_img(random_image_path, target_size=(100, 100))

    random_image_array = img_to_array(random_image)
    random_image_array = random_image_array.reshape(1, 100, 100, 3) / 255.0

    predictions = model.predict(random_image_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = list(train_generator.class_indices.keys())[predicted_class_index]

    if predicted_class in ['Jaguar', 'Lion', 'Leopard', 'Tiger', 'Cheetah']:
        print(f'Model predicts it\'s an animal: {predicted_class}')
    elif predicted_class == 'Humans':
        print('Model predicts it\'s a human with an animal!')
        #winsound.Beep(1000, 500)
    elif predicted_class == 'Endangered_Animals':
        print(f'Model predicts it\'s an endangered animal: {predicted_class}')
        #winsound.Beep(1000, 500)
    else:
        print('No valid prediction for the given image.')

    predictions_data = {
        'image_path': random_image_path,
        'predicted_class': predicted_class,
    }

    return predictions_data if predictions_data else {'image_path': '', 'predicted_class': 'Unknown'}

def preprocess_image(image):
    image = image.resize((100, 100))
    image_array = img_to_array(image)
    image_array = image_array.reshape(1, 100, 100, 3) / 255.0
    return image_array

def make_prediction(image_url):
    response = requests.get(image_url)
    user_image = Image.open(BytesIO(response.content))
    processed_image = preprocess_image(user_image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = list(train_generator.class_indices.keys())[predicted_class_index]
    
    if predicted_class == 'Endangered_Animals':
        message = f'Animal in danger! Contact the department immediately..'
    else:
        message = 'Animal is in danger'
    return predicted_class, message


@app.route('/home')
def index():
    predictions_data = {'image_path': '', 'predicted_class': 'Unknown'}
    return render_template('index.html', predictions_data=predictions_data)

@app.route('/results')
def results():
    predictions_data = classify_random_image()
    return render_template('results.html', predictions_data=predictions_data)

@app.route('/blogs')
def blogs():
    return render_template('blogs.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/human')
def human():
    return render_template('activity.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_image_url = request.form['image_url']
        parsed_url = urlparse(user_image_url)
        if not parsed_url.scheme:
            user_image_url = 'http://' + user_image_url
        predicted_class, message = make_prediction(user_image_url)

        return render_template('output.html', image_url=user_image_url, prediction=predicted_class, message=message)

    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/url', methods=['GET'])
def predict_form():
    return render_template('predict.html')

if __name__ == '__main__':
    
    # model and train_generator
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(7, activation='softmax')  # 7 classes including 'Endangered_Animals'
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    train_generator = get_train_generator()

    app.run(debug=True)

