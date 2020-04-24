# -*- coding: utf-8 -*-
"""
@author: IBartolome and JManchon
"""

import numpy as np
import scipy.signal
from skimage import color
import matplotlib.pyplot as plt


def sobel_h():
    """
        Funcion que devuelve el filtro que detecta las lineas verticales
        
        @return   Numpy Array (3,3) con el filtro 
    """
    return np.array([[1,2,1],
                      [0,0,0],
                      [-1,-2,-1]])
    
def sobel_v():
    """
        Funcion que devuelve el filtro que detecta las lineas horizontales
        
        @return   Numpy Array (3,3) con el filtro
    """
    return sobel_h().T

def sobel_all():
    """
        Funcion que devuelve el filtro que detecta las lineas verticales y horizontales
        
        @return   Numpy Array (3,3) con el filtro
    """
    return sobel_v() + sobel_h()



def gray_img(path="tablero.jpg"):
    """
        Funcion que muestra los filtros en una imagen dada
        
        @param   path   Ruta del fichero
    """
    
    # Lectura de imagen
    img = plt.imread(path)
    
    # Con un solo canal, si fuera RBG, hacerlo cada canal y luego superponer con:
    #    img_res = cv2.merge((np.absolute(img_R),np.absolute(img_G),np.absolute(img_B)))
    img = color.rgb2gray(img)
    
    # Convolucion 2D con los filtros
    img_V = np.absolute(scipy.signal.convolve2d(img, sobel_v(), mode='full'))
    img_H = np.absolute(scipy.signal.convolve2d(img, sobel_h(), mode='full'))
    
    # Normalizacion
    img_V = img_V / np.max(img_V)
    img_H = img_H / np.max(img_H)
    
    # Supresion de negativos (reLU)
    img_V[img_V < 0] = 0
    img_H[img_H < 0] = 0
  
    # Visualizacion 
    fig, ax = plt.subplots( nrows = 1, ncols = 3, figsize = (8, 5), sharex = True , sharey = True )
    plt.gray()
    
    ax[0].imshow (img_V)
    ax[0].axis('off')
    ax[0].set_title ('Sobel Vertical')

    ax[1].imshow ( img_H )
    ax[1].axis ('off')
    ax[1].set_title ('Sobel Horizontal')

    ax[2].imshow (img_V + img_H)
    ax[2].axis('off')
    ax[2].set_title ('Sobel H+V')
    
    

    fig.tight_layout()
    plt.show ( block = False )
    

   
    
if __name__ == "__main__":
    gray_img()
    
    