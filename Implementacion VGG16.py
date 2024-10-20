#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#DatasetFinal

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# Función para cargar los datos desde un CSV y dividirlos
def load_and_split_data(path, test_size=0.2, random_state=2):
    # Cargar el DataFrame desde el CSV
    df = pd.read_csv(path)
    
    # Dividir el DataFrame en conjuntos de entrenamiento y prueba
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    return train_df, test_df

# Rutas a los archivos CSV de cada clase
female_csv_path = r"C:\Users\alvar\OneDrive\Documentos\Universidad\Pruebas\DataAug\Dataset_respaldo\Dataset_solución\female\resized_images_with_labels_female.csv"
male_csv_path = r"C:\Users\alvar\OneDrive\Documentos\Universidad\Pruebas\DataAug\Dataset_respaldo\Dataset_solución\male\resized_images_with_labels_male.csv"

# Cargar y dividir datos para las clases Female y Male
train_female_df, test_female_df = load_and_split_data(female_csv_path)
train_male_df, test_male_df = load_and_split_data(male_csv_path)

# Crear la carpeta de destino si no existe
output_path = r"C:/Users/alvar/OneDrive/Documentos/Universidad/Pruebas/DataAug/Dataset_respaldo/Dataset_final"
os.makedirs(output_path, exist_ok=True)

# Guardar los DataFrames como archivos CSV en la ubicación deseada
train_female_df.to_csv(os.path.join(output_path, 'train_female.csv'), index=False)
test_female_df.to_csv(os.path.join(output_path, 'test_female.csv'), index=False)
train_male_df.to_csv(os.path.join(output_path, 'train_male.csv'), index=False)
test_male_df.to_csv(os.path.join(output_path, 'test_male.csv'), index=False)

# Crear archivo de clases
classes = set(['male', 'female'])

with open(os.path.join(output_path, 'classes.csv'), 'w') as f:
    for i, line in enumerate(sorted(classes)):
        f.write('{},{}\n'.format(line, i))

print("Datos divididos y guardados correctamente.")


# In[ ]:


import os
import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Parámetros
augmented_images_dir = r"C:\Users\alvar\OneDrive\Documentos\Universidad\Pruebas\DataAug\Dataset_respaldo\Augmented_Images"
os.makedirs(augmented_images_dir, exist_ok=True)

# Rutas a los CSV de entrenamiento
train_female_csv = r"C:\Users\alvar\OneDrive\Documentos\Universidad\Pruebas\DataAug\Dataset_respaldo\Dataset_final\train_female.csv"
train_male_csv = r"C:\Users\alvar\OneDrive\Documentos\Universidad\Pruebas\DataAug\Dataset_respaldo\Dataset_final\train_male.csv"

# Cargar los CSV en DataFrames
train_female_df = pd.read_csv(train_female_csv)
train_male_df = pd.read_csv(train_male_csv)

# Combinar los DataFrames de entrenamiento
train_df = pd.concat([train_female_df, train_male_df], ignore_index=True)

# Lista para almacenar las nuevas etiquetas y coordenadas
augmented_data = []

# Función para agregar ruido a la imagen
def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = np.clip(image + noise, 0, 255)  # Asegurarse de que los valores estén en el rango correcto
    return noisy_image

# Procesar cada imagen en el DataFrame y generar aumentaciones
for index, row in train_df.iterrows():
    img_path = row['filename']
    img_class = row['class']
    xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    
    # Cargar la imagen
    img = load_img(img_path)
    img_array = img_to_array(img)

    # Generar 5 imágenes aumentadas por cada imagen original
    for i in range(5):
        augmented_image_path = os.path.join(augmented_images_dir, f"aug_{index}_{i}.jpg")

        if i == 0:  # Desplazamiento horizontal
            shift_x = int(np.random.uniform(-0.1 * img.size[0], 0.1 * img.size[0]))
            xmin += shift_x
            xmax += shift_x
            
        elif i == 1:  # Desplazamiento vertical
            shift_y = int(np.random.uniform(-0.1 * img.size[1], 0.1 * img.size[1]))
            ymin += shift_y
            ymax += shift_y
            
        elif i == 2:  # Agregar ruido
            noisy_image_array = add_noise(img_array)
            Image.fromarray(noisy_image_array.astype(np.uint8)).save(augmented_image_path)

        elif i == 3:  # Cambio a escala de grises
            gray_image_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])  # Convertir a escala de grises
            gray_image_array = np.stack((gray_image_array,) * 3, axis=-1)  # Convertir a formato RGB nuevamente
            
            Image.fromarray(gray_image_array.astype(np.uint8)).save(augmented_image_path)

        elif i == 4:  # Rotación de 90 grados
            rotated_image_array = np.rot90(img_array)  # Rotar la imagen 90 grados
            
            # Ajustar las coordenadas del bounding box tras la rotación
            width = xmax - xmin
            height = ymax - ymin
            
            xmin_new = ymin
            ymin_new = img.size[0] - xmax
            xmax_new = ymin_new + width
            ymax_new = img.size[0] - xmin
            
            xmin, ymin, xmax, ymax = xmin_new, ymin_new, xmax_new, ymax_new

            Image.fromarray(rotated_image_array.astype(np.uint8)).save(augmented_image_path)

        # Asegúrate de que las coordenadas estén dentro de los límites de la imagen.
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img.size[0], xmax)
        ymax = min(img.size[1], ymax)

        # Comprobaciones adicionales para asegurar que las coordenadas son válidas
        if xmin >= xmax or ymin >= ymax:
            print(f"Error en coordenadas: {xmin}, {ymin}, {xmax}, {ymax} para {augmented_image_path}")
            continue  # Saltar esta iteración si hay un error en las coordenadas

        # Agregar la nueva etiqueta y coordenadas a la lista (ajustadas si es necesario)
        augmented_data.append([augmented_image_path, xmin, ymin, xmax, ymax, img_class])

# Crear un DataFrame para las nuevas imágenes y etiquetas
augmented_df = pd.DataFrame(augmented_data, columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])

# Guardar el DataFrame en un nuevo CSV
augmented_df.to_csv(os.path.join(augmented_images_dir, 'augmented_labels.csv'), index=False)

print("Imágenes aumentadas guardadas en:", augmented_images_dir)
print("Etiquetas y coordenadas guardadas en: augmented_labels.csv")


# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix

# Configuración de parámetros
width_shape = 512  # Tamaño de imagen ajustado a 512x512
height_shape = 512
num_classes = 4  # Cambia según el número de clases (hembra_sin_gancho, vientre_hembra, macho_con_gancho, vientre_macho)
epochs = 50
batch_size = 32

# Rutas a los CSV de entrenamiento y prueba
base_dir = r"C:\Users\alvar\OneDrive\Documentos\Universidad\Pruebas\DataAug\Dataset_respaldo\Dataset_final"
csv_files = ['train_female.csv', 'train_male.csv', 'test_female.csv', 'test_male.csv']

# Cargar todas las etiquetas de los CSV en un solo DataFrame
augmented_df = pd.concat([pd.read_csv(os.path.join(base_dir, csv)) for csv in csv_files], ignore_index=True)

# Mapeo de etiquetas a números (one-hot encoding)
labels = {
    'hembra_sin_gancho': 0,
    'vientre_hembra': 1,
    'macho_con_gancho': 2,
    'vientre_macho': 3,
}

# Convertir las etiquetas a números y verificar nulos
augmented_df['label'] = augmented_df['class'].map(labels)

if augmented_df['label'].isnull().any():
    print("Hay etiquetas nulas en el DataFrame de imágenes aumentadas:")
    print(augmented_df[augmented_df['label'].isnull()])

# Convertir a formato one-hot
augmented_labels = np_utils.to_categorical(augmented_df['label'], num_classes=num_classes)

# Cargar imágenes aumentadas en memoria y redimensionar a 512x512
images = []
for filename in augmented_df['filename']:
    try:
        img = cv2.resize(cv2.imread(filename), (width_shape, height_shape))
        images.append(img)
    except Exception as e:
        print(f"Error: No se encontró el archivo {filename}. {e}")

images_array = np.array(images)

# Configuración de generadores de datos para el entrenamiento y la validación
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Crear generadores de datos utilizando flow() en lugar de flow_from_directory()
train_generator = train_datagen.flow(
    x=images_array,
    y=augmented_labels,
    batch_size=batch_size)

validation_generator = valid_datagen.flow(
    x=images_array,
    y=augmented_labels,
    batch_size=batch_size)

# Modelo VGG16 con capas personalizadas
image_input = Input(shape=(height_shape, width_shape, 3))
base_model = VGG16(input_tensor=image_input, include_top=False, weights='imagenet')

# Añadir capas personalizadas al modelo VGG16
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(num_classes, activation='softmax', name='output')(x)

custom_vgg_model = Model(inputs=base_model.input, outputs=output_layer)

for layer in base_model.layers:
    layer.trainable = False

custom_vgg_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
custom_vgg_model.summary()

# Callbacks para monitorear el entrenamiento
tensorboard_callback = TensorBoard(log_dir='./logs')
checkpoint_callback = ModelCheckpoint('best_model_vgg16.h5', save_best_only=True)

# Entrenar el modelo
model_history = custom_vgg_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    steps_per_epoch=len(images_array) // batch_size,
    validation_steps=len(images_array) // batch_size,
    callbacks=[tensorboard_callback, checkpoint_callback]
)

# Guardar el modelo entrenado
custom_vgg_model.save("model_VGG16.h5")

# Graficar resultados del entrenamiento
def plot_training(history):
    plt.figure(figsize=(12, 4))
    
    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

plot_training(model_history)

# Predicción y evaluación del modelo con matriz de confusión

test_data_dir = r"C:\Users\alvar\OneDrive\Documentos\Universidad\Pruebas\DataAug\Dataset_respaldo\Test"
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(width_shape, height_shape),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

predictions = custom_vgg_model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_real = test_generator.classes

conf_mat = confusion_matrix(y_real, y_pred)
plt.figure(figsize=(10,7))
plot_confusion_matrix(conf_mat=conf_mat,
                      figsize=(9,9),
                      class_names=list(labels.keys()),
                      show_normed=True)
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_real, y_pred))


# In[ ]:


import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Ruta al CSV que contiene las etiquetas y coordenadas ajustadas
augmented_labels_csv = r"C:\Users\alvar\OneDrive\Documentos\Universidad\Pruebas\DataAug\Dataset_respaldo\Augmented_Images\augmented_labels.csv"

# Cargar el DataFrame con las etiquetas y coordenadas
augmented_df = pd.read_csv(augmented_labels_csv)

# Número de imágenes a mostrar
num_images_to_show = 5

# Crear una figura para mostrar las imágenes
fig, axs = plt.subplots(1, num_images_to_show, figsize=(15, 5))

# Contador para las imágenes mostradas
count = 0

for i in range(len(augmented_df)):
    # Cargar la imagen
    img_path = augmented_df['filename'].iloc[i]
    
    # Verificar si la imagen existe
    if os.path.exists(img_path):
        img = Image.open(img_path)
        
        # Dibujar la imagen en el subplot
        axs[count].imshow(img)
        axs[count].axis('off')  # Ocultar ejes

        # Obtener las coordenadas del bounding box
        xmin = augmented_df['xmin'].iloc[i]
        ymin = augmented_df['ymin'].iloc[i]
        xmax = augmented_df['xmax'].iloc[i]
        ymax = augmented_df['ymax'].iloc[i]

        # Dibujar el bounding box en rojo
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2, edgecolor='red', facecolor='none')
        axs[count].add_patch(rect)

        # Mostrar la etiqueta de clase en la parte superior de cada imagen.
        axs[count].set_title(augmented_df['class'].iloc[i])
        
        count += 1  # Incrementar contador de imágenes mostradas
        
    # Detenerse si ya se han mostrado suficientes imágenes
    if count >= num_images_to_show:
        break

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




