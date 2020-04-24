import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras import models
from keras import backend as K
import cv2
import warnings

warnings.filterwarnings("ignore")
"""

    Funciones
    
"""


def load_model(img_width,img_height, name):
    """
        Funcion de creacion de la Red
        
        @param   img_width   ancho de la imagen
        @param   img_height  alto de la imagen
        @param   name        nombre del fichero de pesos
        
        @return  modelo 
    """
    model,last_conv_layer = get_model(img_width,img_height)
    model.load_weights(name)
    return model,last_conv_layer


def get_model(img_width,img_height):
    """
        Funcion de creacion de la Red
        
        @param   img_width   ancho de la imagen
        @param   img_height  alto de la imagen
        
        @return  modelo y ultima capa convolucional
    """
    
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    last_conv_layer = Convolution2D(64, 3, 3)
    model.add(last_conv_layer)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    return model, last_conv_layer


def show_heatmap(model,last_conv_layer, imagePath = 'data/cat.jpg', show_layers = False):
    """
    
        Visualizacion
        
    """	
    
    #_Vamos a procesar una imgagen y vemos en que se fija
    img_path = imagePath   #  cat or dog
    
    # Cargamos la imagen y la convertimos en un tensor
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # Como el generador de imagenes lo pusimos con /255, aqui igual
    img_tensor /= 255.
    
    # Its shape is (1, 150, 150, 3)
    # plt.imshow(img_tensor[0])
    # plt.show()
    
    # Ahora vamos a ver la salidas de las capas
    
    # Cogemos las 8 capas que nos interesan:
    layer_outputs = [layer.output for layer in model.layers[:8]]
    # Creamos un modelo con esas capas
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    
    # This will return a list of 8 Numpy arrays:
    # one array per layer activation
    activations = activation_model.predict(img_tensor)
    

    if show_layers:    
        # These are the names of the layers, so can have them as part of our plot
        layer_names = []
        for layer in model.layers[:8]:
            layer_names.append(layer.name)
        
        images_per_row = 16
        
        # Now let's display our feature maps
        for layer_name, layer_activation in zip(layer_names, activations):
            # This is the number of features in the feature map
            n_features = layer_activation.shape[-1]
        
            # The feature map has shape (1, size, size, n_features)
            size = layer_activation.shape[1]
        
            # We will tile the activation channels in this matrix
            n_cols = n_features // images_per_row
            display_grid = np.zeros((size * n_cols, images_per_row * size))
        
            # We'll tile each filter into this big horizontal grid
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[0,
                                                     :, :,
                                                     col * images_per_row + row]
                    # Post-process the feature to make it visually palatable
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size : (col + 1) * size,
                                 row * size : (row + 1) * size] = channel_image
        
            # Display the grid
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            
        plt.show()  
        
        
    """
        Mapa de calor
    """
    
   
    my_model_output = model.output[:, :]
    
    # La ultima capa de la red la devolvemos al crear el modelo, sino seria asi
    # last_conv_layer = model.get_layer('conv2d_9')
    
    # This is the gradient of the class with regard to
    # the output feature map of `last_conv_layer`
    grads = K.gradients(my_model_output, last_conv_layer.output)[0]
    
    # This is a vector of shape (64,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    
    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([img_tensor])
    
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(64):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
       
    
    
    
    # We use cv2 to load the original image
    img = cv2.imread(img_path)
    
    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    
    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 0.4 here is a heatmap intensity factor
    # superimposed_img = heatmap * 0.4 + img
    
    # Save the image to disk
    # cv2.imwrite('test.jpg', superimposed_img)
    
    fig, ax = plt.subplots( nrows = 1, ncols = 3, figsize = (8, 5), sharex = True , sharey = True )
    plt.gray()
    
    ax[0].imshow (img)
    ax[0].axis('off')
    ax[0].set_title ('Imagen Original')

    ax[1].matshow(heatmap)
    ax[1].axis ('off')
    ax[1].set_title ('Mapa de Calor')

    ax[2].imshow (img)
    ax[2].imshow (heatmap, alpha=0.4)
    ax[2].axis('off')
    ax[2].set_title ('Superposicion')
    
    

    fig.tight_layout()
    plt.show ( block = False )
    


"""

    Definiciones
    
"""
if __name__ == "__main__":
    
    # Definimos las dimensiones y donde estan los datos
    img_width, img_height = 150, 150
    train_data_dir        = 'data/train'
    validation_data_dir   = 'data/validation'
    
    # Creamos el modelo y obtenemos la ultima capa convolucional
    model, last_conv_layer = get_model(img_width, img_height)
    
    # Creamos los generadores de datos
    datagen = ImageDataGenerator(rescale=1./255)
        
    train_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=16,
            class_mode='binary')
    
    validation_generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='binary')

#%%  Delimitador para entrenar en Spyder
    """
    
        Entrenamiento y visualizacion
        
    """		  
    nb_epoch              =   5
    nb_train_samples      = 4000
    nb_validation_samples =  400
    model_name            = "model.h5"
    
    # Entrenamiento de la red
    history = model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples)
    		
    # Guardamos la red y la evaluamos
    model.save_weights(model_name)
    model.evaluate_generator(validation_generator, nb_validation_samples)
    
    # Plot del entrenamiento
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

#%% Delimitador para entrenar en Spyder

    show_heatmap(model,last_conv_layer,'data/cat.jpg', show_layers = True)
    show_heatmap(model,last_conv_layer,'data/dog.jpg', show_layers = False)

    
    #  Si queremos cargar el modelo:
    #loaded_model, loaded_last_layer =  load_model(img_width, img_height, "model_50.h5")
    #show_heatmap(loaded_model,loaded_last_layer,'data/cat.jpg', show_layers = True)
    #show_heatmap(loaded_model,loaded_last_layer,'data/dog.jpg', show_layers = False)



    
    
    

