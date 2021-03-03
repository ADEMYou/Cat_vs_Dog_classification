#!/usr/bin/env python
# coding: utf-8

# # Projet Deep Learning : Cat vs Dog

# # ADMYou
# 

# ## Objectifs :
# 
# Ce projet est un projet de vision par ordinateur qui consiste à réaliser une classification binaire d'images de chats et de chiens via des réseaux de neurones.
# 
# Pour ce faire, on dispose d'un jeu de données d'images de chats et de chiens qui provient d'une compétition Kaggle. Bien que nous disposions de 25 000 données, nous allons utiliser qu'une faible partie de ces données, afin de mettre en avant l'apport de la **Data Augmentation** et du **Transfer Learning**.
# 
# Nous suivrons les parties suivantes : 
# 
# - I. Préparation des données
# - II. Réseau de neurones convolutif (CNN)
# - III. CNN + Data Augmentation
# - IV. Transfer Learning (VGG)

# In[19]:


import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, optimizers, utils, regularizers, metrics, preprocessing


# ## I. Préparation des données

# ### Organisation des données

# À partir du dossier "train/" contenant 25 000 images provenant de Kaggle, nous créons (aléatoirement) 3 jeux de données, tous avec des classes parfaitement équilibrées (autant de chiens que de chats) :
# 
# - Un jeu de données de **training** avec 2000 images pour l'entraînement du modèle
# 
# - Un jeu de données de **validation** avec 1000 images pour régler nos hyperparamètres (détecter overfitting/underfitting)
# 
# - Un jeu de données de **test** avec 1000 images pour l'évaluation finale du modèle (meilleur aperçu des performances réelles du modèle, au cas où le modèle sur-apprend sur le jeu de validation)
# 
# Aucune image n'est commune entre ces jeux de données (ensembles disjoints).

# #### Création des répertoires

# In[22]:


# The path to the directory where the original dataset was uncompressed

original_dataset_dir = 'Cat_Dog_original_data'

# The directory where we will store our smaller dataset

new_directories = ['Cat_Dog_small_data', 'Training', 'Validation', 'Test','train_cats_dir', 'train_dogs_dir',
                   'val_cats_dir', 'val_dogs_dir', 'test_cats_dir', 'test_dogs_dir']

parent_directories = ['.', 'Cat_Dog_small_data', 'Cat_Dog_small_data', 'Cat_Dog_small_data',
                      'Cat_Dog_small_data/Training','Cat_Dog_small_data/Training', 'Cat_Dog_small_data/Validation', 
                      'Cat_Dog_small_data/Validation', 'Cat_Dog_small_data/Test', 'Cat_Dog_small_data/Test']

#We check if the directories already exist 
for parent_dir, new_dir in zip(parent_directories, new_directories):
    if new_dir not in os.listdir(parent_dir) :   
        os.mkdir(parent_dir + '/' + new_dir)
    else:
        print(f'Directory "{new_dir}" already exists')


# #### Création des jeux de données

# In[23]:


cat_indexes = list(range(0, 12500))
dog_indexes = list(range(0, 12500))

#Training set
for i in range(1000):    
    j, k = np.random.choice(cat_indexes, size=1)[0], np.random.choice(dog_indexes, size=1)[0]
    shutil.copy2(f'{original_dataset_dir}/cat.{j}.jpg',f'Cat_Dog_small_data/Training/train_cats_dir/cat.{j}.jpg')
    shutil.copy2(f'{original_dataset_dir}/dog.{k}.jpg',f'Cat_Dog_small_data/Training/train_dogs_dir/dog.{k}.jpg')
    cat_indexes.remove(j)
    dog_indexes.remove(k)

#Validation and test sets (no common sample with the training set)
for i in range(500):
    j, k = np.random.choice(cat_indexes, size=1)[0], np.random.choice(dog_indexes, size=1)[0]
    shutil.copy2(f'{original_dataset_dir}/cat.{j}.jpg',f'Cat_Dog_small_data/Validation/val_cats_dir/cat.{j}.jpg')
    shutil.copy2(f'{original_dataset_dir}/dog.{k}.jpg',f'Cat_Dog_small_data/Validation/val_dogs_dir/dog.{k}.jpg')
    cat_indexes.remove(j)
    dog_indexes.remove(k)
    m, n = np.random.choice(cat_indexes, size=1)[0], np.random.choice(dog_indexes, size=1)[0]
    shutil.copy2(f'{original_dataset_dir}/cat.{m}.jpg',f'Cat_Dog_small_data/Test/test_cats_dir/cat.{m}.jpg')
    shutil.copy2(f'{original_dataset_dir}/dog.{n}.jpg',f'Cat_Dog_small_data/Test/test_dogs_dir/dog.{n}.jpg')
    cat_indexes.remove(m)
    dog_indexes.remove(n)


# Vérifions le nombre d'images dans chaque jeu de données : 

# In[9]:


print('total training cat images:', len(os.listdir(parent_directories[4] + '/' + new_directories[4])))


# In[10]:


print('total training dog images:', len(os.listdir(parent_directories[5] + '/' + new_directories[5])))


# In[11]:


print('total validation cat images:', len(os.listdir(parent_directories[6] + '/' + new_directories[6])))


# In[12]:


print('total validation dog images:', len(os.listdir(parent_directories[7] + '/' + new_directories[7])))


# In[13]:


print('total test cat images:', len(os.listdir(parent_directories[8] + '/' + new_directories[8])))


# In[14]:


print('total test dog images:', len(os.listdir(parent_directories[9] + '/' + new_directories[9])))


# On a bien 2000 images d'entraînement, 1000 pour la  validation et 1000 pour le test. Dans chaque jeu de données, il y a le même nombre d'images pour chaque classe : c'est un problème de classification binaire équilibré. L'**accuracy** va être une mesure de performance appropriée.

# ### Pré-traitement des données
# 
# Nous allons dans cette sous-partie, convertir nos images qui sont au format JPEG en tenseurs utilisables par les réseaux de neurones.
# De plus, nous allons normaliser nos données pour que chaque pixel soit compris entre 0 et 1, afin d'accélerer la convergence des algorithmes d'optimisation.
# 
# Nous allons faire ces opérations grâce à la classe **ImageDataGenerator** de Keras.

# In[9]:


from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale= 1./255)
test_datagen = ImageDataGenerator(rescale= 1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    
        parent_directories[1] + '/' + new_directories[1], # This is the target directory
    
        target_size=(150, 150), # All images will be resized to 150x150
        
        batch_size=20,
    
        class_mode='binary') # Since we use binary_crossentropy loss, we need binary labels

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        parent_directories[2] + '/' + new_directories[2],
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary') # Since we use binary_crossentropy loss, we need binary labels

test_generator = test_datagen.flow_from_directory(
                    parent_directories[3] + '/' + new_directories[3],
                    target_size=(150, 150),
                    batch_size=20,
                    class_mode='binary')


# Par ailleurs, on redimensionne les images à (150,150) pixels et on effectue des batchs de 20 images.

# Que signifient les résultats obtenus par les instructions suivantes ?

# ### Réponse :
# **Le code ci-dessous permet d'afficher la taille des batchs du train set : 
# il y a 20 images couleurs de taille 150x150 (pixels), chacune accompagnée de son label (nombre entier 0 ou 1), par batch.**
# 
# Nous avons en plus affiché les 20 images du batch courant du générateur.

# In[10]:


for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    for i in range(data_batch.shape[0]):
        plt.figure()
        plt.title(f'Class {labels_batch[i]}')
        plt.imshow(data_batch[i])
    break


# ## II. CNN

# ### Construction du réseau

# In[11]:


h, w, c = 150, 150, 3


# In[135]:


model = models.Sequential()
#Convolutional layer 1
model.add(layers.Conv2D(filters = 16, 
                        kernel_size=(3,3), 
                        strides = 2, 
                        padding = 'same',
                        activation = 'relu',
                        input_shape = (h, w, c)))

model.add(layers.MaxPooling2D(pool_size = (2,2),
                              strides = 2, 
                              padding = 'valid'))
#Convolutional layer 2
model.add(layers.Conv2D(filters = 32, 
                        kernel_size = (3,3), 
                        strides = 2,
                        padding = 'same',
                        activation = 'relu'))
          
model.add(layers.MaxPooling2D(pool_size = (2,2),
                              strides = 2,
                              padding = 'valid'))
#Fully connected
model.add(layers.Flatten())
model.add(layers.Dense(units = 8, activation = 'relu'))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))


# In[136]:


model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = 'accuracy')


# ### Vérification de l'architecture du réseau

# In[137]:


model.summary()


# On commence par essayer un CNN assez simple (seulement 25 841 paramètres) car on dispose de très peu de données et le réseau va rapidement sur-apprendre (overfitting) s'il est trop complexe.
# 
# - Optimizer : Adam (très performant en général)
# - Loss : Binary cross-entropy (classification binaire)
# - Metric : Accuracy

# On parcourt 100 batchs ( donc 100*20 = 2000 images) pendant un epoch, soit l'intégralité du jeu d'entraînement.
# 
# De plus, on évalue le modèle sur l'ensemble du jeu de validation à chaque epoch.

# In[115]:


history = model.fit(
        train_generator,
        #1 epoch = 100 batchs de 20 images = 2000 images
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        #On évalue sur l'ensemble du jeu de validation (50*20 = 1000)
        validation_steps=50)


# Que fait le code suivant ?

# In[28]:


def learning_curve(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12,8))
    plt.plot(acc,label='Training acc')
    plt.plot(val_acc, label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.figure(figsize=(12,8))
    plt.plot(loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()


# ### Réponse :
# 
# Ce code permet d'afficher les courbes d'accuracy et de loss (coût) sur les jeux de données d'entraînement et de validation (évolution en fonction du nombre d'epochs). La comparaison de ces courbes permettent de détecter les cas d'overfitting ou d'underfitting et de proposer des améliorations du réseau en conséquence.

# In[117]:


learning_curve(history)


# ### Analyse des courbes : 
# 
# On remarque qu'il y un important overfitting : le score sur le training set est bien plus élevé que sur le validation set. L'apprentissage est très bon sur le training set : 0.9957 d'accuracy après 30 epochs tandis qu'il n'est que de 0.7260 sur le validation set.
# 
# Ce phénomène se remarque également avec les courbes de loss.
# 
# On va tester plusieurs méthodes pour lutter contre l'overfitting. Commençons par reprendre le même réseau mais en réalisant des Dropout.

# #### Sauvegarde de notre réseau 

# In[106]:


model.save('CNN_cats_and_dogs_small.h5')


# ### Réseau avec Dropout

# In[31]:


model = models.Sequential()
model.add(layers.Conv2D(filters = 16, 
                        kernel_size=(3,3), 
                        strides = 2, 
                        padding = 'same',
                        activation = 'relu',
                        input_shape = (h, w, c)))

layers.Dropout(0.10)

model.add(layers.MaxPooling2D(pool_size = (2,2),
                              strides = 2, 
                              padding = 'valid'))

model.add(layers.Conv2D(filters = 32, 
                        kernel_size = (3,3), 
                        strides = 2,
                        padding = 'same',
                        activation = 'relu'))
layers.Dropout(0.10)

model.add(layers.MaxPooling2D(pool_size = (2,2),
                              strides = 2,
                              padding = 'valid'))

model.add(layers.Flatten())
model.add(layers.Dense(units = 8, activation = 'relu'))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))


# In[32]:


model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = 'accuracy')


# In[33]:


history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)


# In[34]:


learning_curve(history)


# In[35]:


model = models.Sequential()
model.add(layers.Conv2D(filters = 16, 
                        kernel_size=(3,3), 
                        strides = 2, 
                        padding = 'same',
                        activation = 'relu',
                        input_shape = (h, w, c)))

model.add(layers.Dropout(0.20))

model.add(layers.MaxPooling2D(pool_size = (2,2),
                              strides = 2, 
                              padding = 'valid'))

model.add(layers.Conv2D(filters = 32, 
                        kernel_size = (3,3), 
                        strides = 2,
                        padding = 'same',
                        activation = 'relu'))
model.add(layers.Dropout(0.20))

model.add(layers.MaxPooling2D(pool_size = (2,2),
                              strides = 2,
                              padding = 'valid'))

model.add(layers.Flatten())
model.add(layers.Dense(units = 8, activation = 'relu'))
model.add(layers.Dropout(0.20))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))


# In[36]:


model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = 'accuracy')


# In[37]:


history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)


# In[38]:


learning_curve(history)


# Le Dropout ne semble pas régler le problème. Les performances sur le jeu de validation sont toujours les mêmes : elles stagnent au bout de 5 epochs. Le Dropout a juste altéré les performances du modèle sur le jeu d'entraînement.
# 
# Tentons une **régularisation L2** (weight decay).

# ### Réseau avec régularisation L2

# In[49]:


model = models.Sequential()
model.add(layers.Conv2D(filters = 16, 
                        kernel_size=(3,3), 
                        strides = 2, 
                        padding = 'same',
                        activation = 'relu',
                        input_shape = (h, w, c),
                        kernel_regularizer = 'l2'))

model.add(layers.MaxPooling2D(pool_size = (2,2),
                              strides = 2, 
                              padding = 'valid'))

model.add(layers.Conv2D(filters = 32, 
                        kernel_size = (3,3), 
                        strides = 2,
                        padding = 'same',
                        activation = 'relu',
                        kernel_regularizer = 'l2'))

model.add(layers.MaxPooling2D(pool_size = (2,2),
                              strides = 2,
                              padding = 'valid'))

model.add(layers.Flatten())
model.add(layers.Dense(units = 8, activation = 'relu'))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))


# In[50]:


model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = 'accuracy')


# In[51]:


history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=50)


# In[52]:


learning_curve(history)


# #### **Bilan** : 
# 
# Les méthodes de régularisation (régularisation L2, Dropout) ne permettent pas de réduire l'overfitting. Elles aboutissent seulement à un moins bon score sur le training set.
# 
# Les performances sur le jeu de validation semblent atteindre un pallier au bout de 5-10 epochs : il est impossible de dépasser les 74-75% d'accuracy. Cela est sûrement dû au trop faible nombre de données dont on dispose pour l'entraînement (2000).
# 
# Afin d'acquérir plus de données et donc de variabilité, nous allons effectuer une **Data Augmentation**.
# 

# ## III. Data Augmentation

# À partir des 2000 images du training set, nous allons effectuer plusieurs modifications aléatoires (rotations, cisaillement, crop...) afin d'obtenir plus de données et de variabilité.

# * rotation_range est une valeur en degrés (0-180), une plage à l'intérieur de laquelle on peut faire tourner les  images de façon aléatoire.
# * width_shift et height_shift sont des plages (en fraction de la largeur ou de la hauteur totale) à  l'intérieur desquelles il est possible de déplacer aléatoirement des images verticalement ou horizontalement.
# * shear_range est une plage permettant d'appliquer de manière aléatoire des transformations de cisaillement.
# * zoom_range permet de zoomer de manière aléatoire à l'intérieur des images.
# * horizontal_flip est pour retourner de manière aléatoire la moitié des images horizontalement -- pertinent lorsqu'il n'y a pas d'hypothèse d'asymétrie horizontale (par exemple, les images du monde réel).
# * fill_mode est la stratégie utilisée pour remplir les pixels nouvellement créés,  qui peuvent apparaître après une rotation ou un décalage largeur/hauteur.
# 
# 

# In[102]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        parent_directories[1] + '/' + new_directories[1],
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        parent_directories[2] + '/' + new_directories[2],
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
                    parent_directories[3] + '/' + new_directories[3],
                    target_size=(150, 150),
                    batch_size=20,
                    class_mode='binary')


# In[66]:


model = models.Sequential()
model.add(layers.Conv2D(filters = 16, 
                        kernel_size=(3,3), 
                        strides = 2, 
                        padding = 'same',
                        activation = 'relu',
                        input_shape = (h, w, c)))

model.add(layers.MaxPooling2D(pool_size = (2,2),
                              strides = 2, 
                              padding = 'valid'))

model.add(layers.Conv2D(filters = 32, 
                        kernel_size = (3,3), 
                        strides = 2,
                        padding = 'same',
                        activation = 'relu'))
          
model.add(layers.MaxPooling2D(pool_size = (2,2),
                              strides = 2,
                              padding = 'valid'))
model.add(layers.Flatten())
model.add(layers.Dense(units = 8, activation = 'relu'))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))


# In[67]:


model.compile(optimizer ='Adam', loss ='binary_crossentropy', metrics = 'accuracy')


# In[68]:


history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)


# In[69]:


learning_curve(history)


# In[70]:


model.save('CNN_Data_Aug_cats_and_dogs_small.h5')


# Discuter et analyser les courbes que vous avez obtenues et les comparer aux courbes que vous avez obtenues sans l'augmentation des données.

# ### Réponse : 
# 
# - On constate que l'écart entre les courbes d'entraînement et de validation a significativement été réduit. L'écart est maintenant faible : après 100 epochs, l'accuracy est de 0.7776 pour le training et 0.7760 pour la validation (sans **data augmentation** : 0.9957 contre 0.7260).
# 
# 
# - Cela signifie que notre problème d'overfitting a été résolu : le modèle généralise très bien ce qu'il a appris sur le jeu d'entraînement.
# 
# 
# - Néanmoins, l'apprentissage du modèle sur le training set est moins bon (underfitting). Sans **data augmentation**, le score sur le train était de 0.9957 après 100 epochs, contre 0.7776 ici. Tentons de complexifier le modèle.

# In[147]:


model = models.Sequential()
model.add(layers.Conv2D(filters = 16, 
                        kernel_size=(3,3), 
                        strides = 2, 
                        padding = 'same',
                        activation = 'relu',
                        input_shape = (h, w, c)))

model.add(layers.MaxPooling2D(pool_size = (2,2),
                              strides = 2, 
                              padding = 'valid'))

model.add(layers.Conv2D(filters = 32, 
                        kernel_size = (3,3), 
                        strides = 2,
                        padding = 'same',
                        activation = 'relu'))
          
model.add(layers.MaxPooling2D(pool_size = (2,2),
                              strides = 2,
                              padding = 'valid'))

model.add(layers.Flatten())
model.add(layers.Dense(units = 16, activation = 'relu'))
model.add(layers.Dense(units = 16, activation = 'relu'))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))


# In[148]:


model.compile(optimizer = 'Adam', 
              loss ='binary_crossentropy', 
              metrics = 'accuracy')


# In[149]:


history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)


# In[150]:


learning_curve(history)


# In[151]:


print(f'More complex model with Data Augmentation : {model.evaluate(test_generator)}')


# ### Bilan 
# 
# Avec ce modèle plus complexe (nous avons testé plusieurs modèles plus complexes, qui sont similaires à celui-ci en termes de performances), on obtient des résultats très similaires au modèle précédent : il n'y a pas de réelle amélioration.
# 
# Il n'y a donc pas d'overfitting mais un biais important (underfitting) : on est en dessous des 80% d'accuracy sur les jeux d'entraînement et de validation.
# 
# Afin d'avoir un modèle sans overfitting et qui s'approche des 100% d'accuracy, nous allons tenter une approche via **Transfer Learning**.

# ## IV. Transfer Learning par Fine-tuning

# Malgré l'augmentation de données, le nombre de données dont nous disposons pour entraîner le modèle reste faible. Par conséquent, la variabilité des données que nous fournissons au modèle est également faible et ne reflète pas la grande variabilité des images de chats et chiens.
# 
# Nous allons mettre en place une approche de **Transfer learning** en utilisant un CNN bien connu pour la classification d'images : **VGG16**. Ce réseau, dont l'architecture est présentée ci-après, a été entraîné sur plus d'un million d'images réparties en 1000 classes (ImageNet).

# On importe via Keras **VGG16** avec tous ses poids, mais sans le réseau Dense (Fully connected) utilisé pour la classification car nous allons développer notre propre réseau Dense pour notre classification binaire.

# In[158]:


from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))


# In[159]:


conv_base.summary()


# À votre avis, que fait le code suivant ?

# In[166]:


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# ### Réponse : 
# 
# Ce code crée le réseau de neurones qu'on désire à partir du réseau VGG16 pré-entraîné et d'un nouveau réseau Dense (Fully connected) qu'on va entraîner afin d'effectuer la classification binaire. En effet, nous ne pouvons pas conserver la partie Dense de VGG16 car celle-ci est conçue pour un problème de classification différent (à 1000 classes).

# In[167]:


model.compile(optimizer = 'Adam', 
              loss ='binary_crossentropy', 
              metrics = 'accuracy')


# In[168]:


model.summary()


# Combien de paramètres sont utilisés par le réseau VGG16 ?

# ### Réponse : 
# 
# Le réseau VGG16 possède **14 714 688** paramètres.

# Étant donné que nous disposons d'un faible nombre de données, nous allons entraîner uniquement la partie Dense du réseau. Les poids du réseau VGG16 ne seront pas modifiés.

# In[169]:


conv_base.trainable = False


# Que fait le code suivant ? Compléter ce code.

# In[170]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        parent_directories[1] + '/' + new_directories[1],
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        parent_directories[2] + '/' + new_directories[2],
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
                    parent_directories[3] + '/' + new_directories[3],
                    target_size=(150, 150),
                    batch_size=20,
                    class_mode='binary')


# ### Réponse :
# 
# Ce code permet, via la classe ImageDataGenerator, d'importer et de pré-traiter efficacement et rapidement nos images : 
# 
# - Il permet d'effectuer de l'augmentation de données, grâce à l'objet ImageDataGenerator, en appliquant de façon aléatoire à nos images les transformations précisées en paramètres (pour le training set)
# 
# 
# - Il permet, toujours grâce à ImageDataGenerator, et sa méthode flow_from_directory, de lire les images sur le disque, au format JPEG, et de les convertir en tenseurs, utilisables par les réseaux de neurones.
# 
# 
# - Il permet également de pré-traiter nos images : normalisation, redimensionnement des images et création de batchs.

# ### Fine-tuning
# 
# Nous allons dégeler le bloc 5 de convolution et le ré-entraîner afin d'en tirer des caractéristiques plus adaptées à notre modèle (fine-tuning).
# 
# ![fine-tuning VGG16](https://s3.amazonaws.com/book.keras.io/img/ch5/vgg16_fine_tuning.png)

# À votre avis, que fait le code suivant ?

# In[174]:


conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


# ### Réponse : 
# 
# Ce code permet de dégeler toutes les couches à partir de la 5e couche 'block5_conv1', qui est la première couche de convolution du bloc 5.
# 

# In[175]:


conv_base.summary()


# In[176]:


model.summary()


# Combien de paramètres sont utilisés par ce nouveau réseau ?

# ### Réponse : 
# 
# Désormais, **7 079 424** paramètres du réseau VGG16, et **9 177 089** en tout pour notre modèle complet (avec le réseau Dense) vont être utilisés (modifiés pendant l'entraînement).

# Donner quelques argumuments, pourquoi il n'est pas judicieux d'utiliser toutes les couches de VGG16 ?

# ### Réponse : 
# 
# Il n'est pas judicieux de ré-entraîner toutes les couches du réseau VGG16 car : 
# 
# - VGG a été entraîné pour de la classification d'images. Notre problème est également une classification d'images. Les premières couches de VGG extraient des caractéristiques de bas niveau (arêtes, coins, blobs...?), probablement communes à toutes les images, et sont donc des caractéristiques pertinentes pour notre problème.
# 
# 
# - Comme l'initialisation des nouvelles couches denses est faite aléatoirement, d'importantes mises à jour des poids vont avoir lieu dans tout le réseau, détruisant alors tout l'apprentissage qui a été fait sur le réseau VGG sur le million d'images (toutes les caractéristiques intéressantes vont être détruites). Comme nous disposons de peu de données, nous n'arriverons pas à retrouver d'aussi bonnes valeurs pour les poids des premières couches.
# 
# 
# - En ne ré-entraînant pas toutes les couches, il y a moins de risques d'overfitting. En effet, le réseau que nous développons à partir de VGG est très complexe et dispose de beaucoup de paramètres. Avec le peu de données dont nous disposons, le réseau pourrait sur-apprendre et perdre sa capacité de généralisation. En utilisant les poids de VGG, le réseau pourra tirer profit de caractéristiques qui ne sont pas trop spécifiques au jeu d'entraînement sur lequel nous faisons l'apprentissage.
# 
# 
# - VGG étant un réseau profond, l'apprentissage est coûteux en termes de puissance et temps de calcul.

# In[177]:


history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)


# In[179]:


learning_curve(history)


# In[180]:


model.save('Transfer_learning_cats_and_dogs_small.h5')


# In[181]:


test_loss, test_acc = model.evaluate(test_generator, steps=50)
print('Test accuracy :', test_acc)


# Quelles sont les performances sur la base test ?

# ### Réponse 
# 
# Sur le jeu de test, nous obtenons **0.8980** d'accuracy, ce qui est similaire aux performances sur le jeu de validation. Nous pouvons donc affirmer que sur des jeux de données futurs, le modèle aura également environ 90% d'accuracy.
# 
# De plus, le coût (loss) est de 0.3028.

# Commenter les nouvelles performances et les comparer aux résultats précédents.

# ### Réponse : 
# 
# - Après 50 epochs, les écarts entre les courbes d'entraînement et de validation se creusent : les performances sont meilleures sur le jeu d'entraînement. Après 100 epochs, nous obtenons sur le jeu d'entraînement **0.9492** d'accuracy contre **0.9010** sur le jeu de validation. Il y a donc un problème d'**overfitting**. 
# 
# 
# - Toutefois, les performances obtenues sur les jeux d'entraînement et de validation sont nettement supérieures à celles que nous avons obtenues précédemment avec notre propre CNN et de l'augmentation de données (sur le jeu de test : 0.7950 contre 0.8980 avec le transfer learning).
# 
# 
# - Afin de réduire l'overfitting, nous pouvons refaire du transfer learning mais en fine-tunant uniquement les couches denses, ou bien tenter des méthodes de régularisation (Dropout, régularisation L2, early stopping...).

# ## Bilan : 
# 
# À travers ce projet, nous avons pu découvrir et expérimenter des méthodes très utilisées en Deep learning et vision par ordinateur lorsqu'on dispose d'un petit jeu de données. 
# 
# En effet, les réseaux de neurones nécessitent généralement beaucoup de données (sinon ils vont rapidement sur-apprendre étant donné leur nombre élevé de paramètres), mais il n'est pas toujours possible d'en avoir autant (données difficiles à étiqueter, données sensibles...). Toutefois, ce n'est pas une fatalité et nous pouvons obtenir de bons résultats avec peu de données.
# 
# L'augmentation de données (Data augmentation) peut permettre de lutter efficacement contre l'overfitting, comme nous avons pu le voir dans ce projet. Néanmoins, celle-ci a ses limites car elle n'ajoute que peu de variabilité dans les données, et elle ne permettra pas d'obtenir de largement meilleures performances sur le jeu de validation.
# 
# Par contre, le transfer learning, lui, est beaucoup plus efficace. C'est véritablement la première méthode à laquelle il faut songer lorsqu'on dispose d'un petit jeu de données. En utilisant un réseau pré-entraîné pour une tâche similaire à la nôtre, nous avons obtenu de largement meilleurs résultats que précédemment, car nous avons tirer profit du million d'images qui a servi à l'apprentissage de VGG.
