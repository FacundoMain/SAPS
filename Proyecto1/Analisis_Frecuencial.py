# -*- coding: utf-8 -*-
"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Script ejemplificando el uso de la función provista para levanta los registros
almacenados en .csv, el maneho de las señales y su graficación. 

Autor: Albano Peñalva
Fecha: Marzo 2022
"""

# Librerías
from scipy import fft
import process_data
import numpy as np
import matplotlib.pyplot as plt

#%% Lectura del dataset

FS = 500 # Frecuencia de muestre: 500Hz
T = 3    # Tiempo total de cada registro: 3 segundos

folder = 'dataset' # Carpeta donde se almacenan los .csv

x, y, z, classmap = process_data.process_data(FS, T, folder)

#%% Graficación

ts = 1 / FS                     # tiempo de muestreo
N = FS*T                        # número de muestras en cada regsitro
t = np.linspace(0, N * ts, N)   # vector de tiempos

# Se crea un arreglo de gráficas, con tres columnas de gráficas 
# (correspondientes a cada eje) y tantos renglones como gesto distintos.
fig, axes = plt.subplots(len(classmap), 3, figsize=(20, 20))
fig.subplots_adjust(hspace=0.5)

# Se recorren y grafican todos los registros
trial_num = 0
amplitudMax_x = 0
amplitudMax_y = 0
amplitudMax_z = 0
fxMax = 0
fyMax = 0
fzMax = 0
for gesture_name in classmap:                           # Se recorre cada gesto
    for capture in range(int(len(x))):                  # Se recorre cada renglón de las matrices
        if (x[capture, N] == gesture_name):
            
            x[capture, 0:N] = x[capture, 0:N]*0.3+1.65  #Señal en volts
            y[capture, 0:N] = y[capture, 0:N]*0.3+1.65
            z[capture, 0:N] = z[capture, 0:N]*0.3+1.65
             # Si en el último elemento se detecta la etiqueta correspondiente
            fx, senial_x_fft = process_data.calculo_fft(x[capture, 0:N], FS) 
            fy, senial_y_fft = process_data.calculo_fft(y[capture, 0:N], FS)
            fz, senial_z_fft = process_data.calculo_fft(z[capture, 0:N], FS)
            
            # Mayor voltaje frecuencia.
            
            FS_filtro = 80
            FS_filtro_2 = FS_filtro/2
            
           
            amplitudx_aux = np.max ( senial_x_fft [np.where (fx >= FS_filtro_2)] )
            fx_aux = fx [np.argmax( senial_x_fft [np.where (fx >= FS_filtro_2)])] + FS_filtro_2
            if (amplitudx_aux > amplitudMax_x ):
                amplitudMax_x = amplitudx_aux 
                fxMax = fx_aux
            valorFREC65 = senial_x_fft [np.where(fx == 65)]
                
            amplitudy_aux = np.max ( senial_y_fft [np.where (fy >= FS_filtro_2)])
            fy_aux = fy [np.argmax( senial_y_fft [np.where (fy >= FS_filtro_2)])] + FS_filtro_2
            if (amplitudy_aux > amplitudMax_y ):
                amplitudMax_y = amplitudy_aux 
                fyMax = fy_aux
                
                
            amplitudz_aux = np.max ( senial_z_fft [np.where (fz >= FS_filtro_2)])
            fz_aux = fz [np.argmax( senial_z_fft [np.where (fz >= FS_filtro_2)])] + FS_filtro_2
            if (amplitudz_aux > amplitudMax_z ):
                amplitudMax_z = amplitudz_aux
                fzMax = fz_aux 
        
            
            # Se grafica la señal en los tres ejes
            axes[gesture_name][0].plot(fx, senial_x_fft, label="Trial {}".format(trial_num))
            axes[gesture_name][1].plot(fy, senial_y_fft, label="Trial {}".format(trial_num))
            axes[gesture_name][2].plot(fz[0:300], senial_z_fft[0:300], label="Trial {}".format(trial_num))
            trial_num = trial_num + 1
    
    
# Se le da formato a los ejes de cada gráfica
    axes[gesture_name][0].set_title(classmap[gesture_name] + " (Aceleración X)")
    axes[gesture_name][0].grid()
    axes[gesture_name][0].legend(fontsize=6, loc='upper right');
    axes[gesture_name][0].set_xlabel('Frecuencia [Hz]', fontsize=10)
    axes[gesture_name][0].set_ylabel('Voltaje [V]', fontsize=10)
    axes[gesture_name][0].set_ylim(0, 0.15)
    
    axes[gesture_name][1].set_title(classmap[gesture_name] + " (Aceleración Y)")
    axes[gesture_name][1].grid()
    axes[gesture_name][1].legend(fontsize=6, loc='upper right');
    axes[gesture_name][1].set_xlabel('Frecuencia [Hz]', fontsize=10)
    axes[gesture_name][1].set_ylabel('Voltaje [V]', fontsize=10)
    axes[gesture_name][1].set_ylim(0, 0.15)
    
    axes[gesture_name][2].set_title(classmap[gesture_name] + " (Aceleración Z)")
    axes[gesture_name][2].grid()
    axes[gesture_name][2].legend(fontsize=6, loc='upper right');
    axes[gesture_name][2].set_xlabel('Frecuencia [Hz]', fontsize=10)
    axes[gesture_name][2].set_ylabel('Voltaje [V]', fontsize=10)
    axes[gesture_name][2].set_ylim(0, 0.15)
    
plt.tight_layout()
plt.show()

print ("Amplitud MAX en X: ", amplitudMax_x )
print ("Amplitud MAX en Y: ", amplitudMax_y )
print ("Amplitud MAX en Z: ", amplitudMax_z )
print ("Frecuencia MAX en X: ", fxMax)
print ("Frecuencia MAX en Y: " , fyMax)
print ("Frecuencia max en Z: ", fzMax)
