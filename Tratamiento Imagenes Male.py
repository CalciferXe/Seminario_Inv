#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Recortar imagen
from PIL import Image
import os

# Ruta de la carpeta con las imágenes
input_folder = "C:/Users/alvar/OneDrive/Documentos/Universidad/Pruebas/DataAug/Dataset_respaldo/Dataset_respaldo/Male"  # Cambia esto a tu ruta

# Define el área a recortar (izquierda, arriba, derecha, abajo)
crop_area = (1130, 1400, 4760, 3140)  # Ajusta estos valores según sea necesario

def crop_images_in_folder(folder):
    # Verifica si la carpeta existe
    if not os.path.exists(folder):
        print(f"La carpeta {folder} no existe.")
        return

    # Itera sobre cada archivo en la carpeta
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Formatos de imagen
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    # Verifica si la imagen es lo suficientemente grande para el recorte
                    if img.size[0] < crop_area[2] or img.size[1] < crop_area[3]:
                        print(f"Imagen {filename} es demasiado pequeña para recortar.")
                        continue
                    
                    # Recorta la imagen
                    cropped_img = img.crop(crop_area)
                    # Guarda la imagen recortada
                    cropped_img.save(os.path.join(folder, f"cropped_{filename}"))
                    print(f"Cropped and saved: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Ejecuta la función
crop_images_in_folder(input_folder)


# In[ ]:


#Reescalar imagen
from PIL import Image
import os

# Ruta de la carpeta con las imágenes
input_folder =  "C:/Users/alvar/OneDrive/Documentos/Universidad/Pruebas/DataAug/Dataset_respaldo/Dataset_respaldo/Male"   # Cambia esto a tu ruta
output_folder = "C:/Users/alvar/OneDrive/Documentos/Universidad/Pruebas/DataAug/Dataset_respaldo/Dataset_rescalado/Male"  # Carpeta de salida

# Crea la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

def resize_images_in_folder(folder, output_folder, target_size):
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Formatos de imagen
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    # Calcula el nuevo tamaño manteniendo la relación de aspecto
                    img.thumbnail((target_size, target_size))
                    # Guarda la imagen reescalada en la carpeta de salida
                    resized_img_path = os.path.join(output_folder, filename)
                    img.save(resized_img_path)
                    print(f"Reescalada y guardada: {filename}")
            except Exception as e:
                print(f"Error procesando {filename}: {e}")

# Ejecuta la función con un tamaño objetivo de 512 píxeles
resize_images_in_folder(input_folder, output_folder, 512)


# In[ ]:


#Etiquetar XML a CSV 
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_images_with_labels_csv(path):
    data = []  # Lista para almacenar los datos

    for xml_file in glob.glob(os.path.join(path, '*.xml')):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Obtener el nombre del archivo de la imagen
            filename = root.find('filename').text
            image_path = root.find('path').text
            
            # Verificar si se encontraron los valores esperados
            if filename is None or image_path is None:
                print(f'Advertencia: No se encontró <filename> o <path> en {xml_file}')
                continue
            
            # Iterar sobre cada objeto en el XML
            for member in root.findall('object'):
                bndbox = member.find('bndbox')
                if bndbox is not None:
                    value = (
                        image_path,  # Ruta completa de la imagen
                        int(bndbox.find('xmin').text),
                        int(bndbox.find('ymin').text),
                        int(bndbox.find('xmax').text),
                        int(bndbox.find('ymax').text),
                        member.find('name').text  # Nombre de la etiqueta
                    )
                    data.append(value)
                else:
                    print(f'Advertencia: No se encontró <bndbox> en {xml_file}')

        except Exception as e:
            print(f'Error al procesar {xml_file}: {e}')

    # Crear un DataFrame con los datos
    column_names = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    images_df = pd.DataFrame(data, columns=column_names)

    return images_df

# Cambia esta ruta a la carpeta donde se encuentran tus archivos XML
xml_folder_path = r"C:\Users\alvar\OneDrive\Documentos\Universidad\Pruebas\DataAug\Dataset_respaldo\Dataset_rescalado\Male"

# Llama a la función con la nueva ruta
images_df = xml_to_images_with_labels_csv(xml_folder_path)

# Guarda el DataFrame como un archivo CSV en la misma carpeta que los XML
output_csv_path = os.path.join(xml_folder_path, 'images_with_labels_male.csv')  # Guardar en la misma carpeta
images_df.to_csv(output_csv_path, index=False)

# Verificar si el archivo CSV se guardó correctamente
if os.path.exists(output_csv_path):
    print('Archivo CSV con imágenes y etiquetas guardado en:', output_csv_path)
    print(f'Número de entradas guardadas: {len(images_df)}')
else:
    print('Error: El archivo CSV no se pudo guardar.')


# In[ ]:


#Hacer img unicas con labels
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Función para mostrar objetos en la imagen
def showObjects(image_df):
    img_path = image_df['filename'].iloc[0]  # Obtener la ruta de la imagen

    # Cargar la imagen
    image = io.imread(img_path)
    
    # Crear figura y ejes
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Filtrar el DataFrame para obtener solo los objetos de esta imagen
    objects_in_image = images_df[images_df['filename'] == img_path]

    # Dibujar un rectángulo para cada objeto
    for _, obj in objects_in_image.iterrows():
        rect = patches.Rectangle(
            (obj.xmin, obj.ymin),
            obj.xmax - obj.xmin,
            obj.ymax - obj.ymin,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(obj.xmin, obj.ymin, obj['class'], color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.title(f"Imagen: {img_path}")  # Título con el nombre de la imagen
    plt.show()

# Crear un DataFrame con imágenes únicas (esto debe hacerse después de crear images_df)
unique_images_df = images_df[['filename']].drop_duplicates()

# Mostrar las imágenes únicas con sus objetos etiquetados
for _, unique_img in unique_images_df.iterrows():
    showObjects(unique_img.to_frame().T)  # Convertir la serie a DataFrame


# In[ ]:


#Mostrar cantidad de entradas
print("Número de objetos en unique_img.to_frame:", len(unique_images_df ))
# Cambia el índice para mostrar diferentes imágenes


# In[1]:


#Transformando imagenes a 512x512 para evaluación
import os
import pandas as pd
from PIL import Image

# Ruta a la carpeta donde están las imágenes y el archivo CSV
folder_path = r"C:\Users\alvar\OneDrive\Documentos\Universidad\Pruebas\DataAug\Dataset_respaldo\Dataset_rescalado\Male"
csv_path = os.path.join(folder_path, 'images_with_labels_male.csv')

# Cargar el CSV que contiene las etiquetas
annotations = pd.read_csv(csv_path)

# Crear una carpeta para guardar las imágenes redimensionadas en la nueva ubicación
output_folder = r"C:\Users\alvar\OneDrive\Documentos\Universidad\Pruebas\DataAug\Dataset_respaldo\Dataset_solución\male"
os.makedirs(output_folder, exist_ok=True)

# Dimensiones deseadas
new_size = (512, 512)

# Redimensionar imágenes y ajustar etiquetas
for index, row in annotations.iterrows():
    image_path = row['filename']  # Asumiendo que la columna se llama 'filename'
    
    # Verificar si la imagen existe antes de intentar abrirla
    if not os.path.exists(image_path):
        print(f"Advertencia: La imagen {image_path} no se encuentra. Se omitirá.")
        continue
    
    img = Image.open(image_path)
    
    # Redimensionar la imagen
    img_resized = img.resize(new_size)
    
    # Ajustar coordenadas de bounding box
    width_ratio = new_size[0] / img.width
    height_ratio = new_size[1] / img.height
    
    annotations.at[index, 'xmin'] *= width_ratio
    annotations.at[index, 'xmax'] *= width_ratio
    annotations.at[index, 'ymin'] *= height_ratio
    annotations.at[index, 'ymax'] *= height_ratio
    
    # Guardar la imagen redimensionada en la nueva carpeta
    resized_image_path = os.path.join(output_folder, os.path.basename(image_path))
    img_resized.save(resized_image_path)

# Guardar el nuevo CSV con las coordenadas ajustadas en la nueva ubicación
resized_csv_path = os.path.join(output_folder, 'resized_images_with_labels_male.csv')
annotations.to_csv(resized_csv_path, index=False)

print("Imágenes redimensionadas y etiquetas ajustadas guardadas correctamente.")


# In[2]:


#Comprobar que las etiquetas estan bien ubicadas
import os
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Cargar el DataFrame que contiene las etiquetas (asegúrate de cargarlo correctamente)
folder_path = r"C:\Users\alvar\OneDrive\Documentos\Universidad\Pruebas\DataAug\Dataset_respaldo\Dataset_solución\male"
csv_path = os.path.join(folder_path, 'resized_images_with_labels_male.csv')
images_df = pd.read_csv(csv_path)

# Función para mostrar objetos en la imagen
def showObjects(image_df):
    img_path = image_df['filename'].iloc[0]  # Obtener la ruta de la imagen

    # Cargar la imagen
    image = io.imread(img_path)
    
    # Crear figura y ejes
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Filtrar el DataFrame para obtener solo los objetos de esta imagen
    objects_in_image = images_df[images_df['filename'] == img_path]

    # Dibujar un rectángulo para cada objeto
    for _, obj in objects_in_image.iterrows():
        rect = patches.Rectangle(
            (obj.xmin, obj.ymin),
            obj.xmax - obj.xmin,
            obj.ymax - obj.ymin,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(obj.xmin, obj.ymin, obj['class'], color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.title(f"Imagen: {img_path}")  # Título con el nombre de la imagen
    plt.show()

# Crear un DataFrame con imágenes únicas (esto debe hacerse después de crear images_df)
unique_images_df = images_df[['filename']].drop_duplicates()

# Mostrar las imágenes únicas con sus objetos etiquetados
for _, unique_img in unique_images_df.iterrows():
    showObjects(unique_img.to_frame().T)  # Convertir la serie a DataFrame


# In[ ]:




