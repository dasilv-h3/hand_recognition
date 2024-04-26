from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

train_data_dir = 'gestures'
weights_path = './vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Paramètres pour la préparation des données
img_width, img_height = 150, 150  # Taille des images
batch_size = 32

# Générateur d'images pour l'entraînement
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

base_model = VGG16(weights=weights_path, include_top=False, input_shape=(img_width, img_height, 3))

for layer in base_model.layers:
    layer.trainable = False


model = Sequential() #NN vide
# Ajouter la base_model (VGG16) en tant que couche de base
model.add(base_model)
model.add(Flatten())
model.add(Dense(500, activation="relu", input_shape=(img_width * img_height * 3,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd"
)

model.summary()

model.fit(train_generator, epochs=50)

model.save('model.h5')