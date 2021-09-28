#!/usr/bin/env python

"""
Este programa implementa un freno de emergencia para evitar accidentes en Duckietown.
"""
import math
import sys
import argparse
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
import numpy as np
import cv2


def mov_duckiebot(key):
    # La acción de Duckiebot consiste en dos valores:
    # velocidad lineal y velocidad de giro
    actions = {ord('w'): np.array([1.0, 0.0]),
               ord('s'): np.array([-1.0, 0.0]),
               ord('a'): np.array([0.0, 1.0]),
               ord('d'): np.array([0.0, -1.0]),
               ord('q'): np.array([0.3, 1.0]),
               ord('e'): np.array([0.3, -1.0])
               }

    action = actions.get(key, np.array([0.0, 0.0]))
    return action


def det_duckie(obs):
    lower_yellow = np.array([10, 200, 150])
    upper_yellow = np.array([35, 255, 255])
    min_area = 2500

    image_out = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(image_out, lower_yellow, upper_yellow)
    image_out_2 = cv2.bitwise_and(obs, obs, mask=mask)
    kernel = np.ones((5, 5), np.uint8)
    image_out_3 = cv2.erode(image_out_2, kernel, iterations=1)
    image_out_4 = cv2.dilate(image_out_3, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    dets = list()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        AREA = w * h
        if AREA > min_area:
            # En lugar de dibujar, se agrega a la lista
            dets.append((x, y, w, h))

    return dets


def draw_dets(obs, dets):
    for d in dets:
        x1, y1 = d[0], d[1]
        x2 = x1 + d[2]
        y2 = y1 + d[3]
        cv2.rectangle(obs, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    return obs


def red_alert(obs):
    red_img = np.zeros(obs.shape, dtype=np.uint8)
    red_img[:, :, 0] = 255
    blend = cv2.addWeighted(obs, 0.5, red_img, 0.5, 0)

    return blend


if __name__ == '__main__':

    # Se leen los argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="Duckietown-udem1-v1")
    parser.add_argument('--map-name', default='udem1')
    args = parser.parse_args()

    # Definición del environment
    if args.env_name and args.env_name.find('Duckietown') != -1:
        env = DuckietownEnv(
            map_name=args.map_name,
            domain_rand=False,
        )
    else:
        env = gym.make(args.env_name)

    # Se reinicia el environment
    env.reset()

    # Inicialmente no hay alerta
    alert = False

    # Posición del pato en el mapa (fija)
    duck_pos = np.array([2, 0, 2])

    # Constante que se debe calcular
    C = 66  # f * dr (f es constante, dr es conocido)

    while True:

        # Captura la tecla que está siendo apretada y almacena su valor en key
        key = cv2.waitKey(0)
        # Si la tecla es Esc, se sale del loop y termina el programa
        if key == 27:
            break

        # Se define la acción dada la tecla presionada
        action = mov_duckiebot(key)

        # Si hay alerta evitar que el Duckiebot avance
        if alert:
            if key == ord('w'):
                action = np.array([0.0, 0.0])
            else:
                pass

            
        # Se ejecuta la acción definida anteriormente y se retorna la observación (obs),
        # la evaluación (reward), etc
        obs, reward, done, info = env.step(action)

        # Detección de patos, retorna lista de detecciones
        dets = det_duckie(obs)
        # Dibuja las detecciones
        obs = draw_dets(obs, dets)
        # Obtener posición del duckiebot
        dbot_pos = env.cur_pos


        # Calcular distancia real entre posición del duckiebot y pato
        # esta distancia se utiliza para calcular la constante
        dist = math.sqrt((dbot_pos[0]-duck_pos[0])**2+(dbot_pos[1]-duck_pos[1])**2+(dbot_pos[2]-duck_pos[2])**2)

        # La alerta se desactiva (opción por defecto)
        alert = False

        for d in dets:
            # Alto de la detección en pixeles
            p = d[3]
            # La aproximación se calcula según la fórmula mostrada en la capacitación
            d_aprox = C/p

            # Muestra información relevante
            print('p:', p)
            print('Da:', d_aprox)
            print('Dr:', dist)

            # Si la distancia es muy pequeña activa alerta
            if d_aprox < 0.3:
                # Activar alarma
                alert = True
                # Muestra ventana en rojo
                obs = red_alert(obs)


        # Se muestra en una ventana llamada "patos" la observación del simulador
        cv2.imshow('patos', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    # Se cierra el environment y termina el programa
    env.close()
