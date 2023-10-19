import numpy as np
import os
import pandas as pd
from skimage import color, feature, io, morphology
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.svm import SVC
from tqdm.auto import tqdm


def preprocesamiento(img, selem):
    img = morphology.opening(img, selem)
    return img


class DataGenerator:
    """
    Clase que genera un lector de datos para las imágenes de entrenamiento,
    prueba y validación
    """

    def __init__(self, data_dirs, grayscale=True, preprocessing=False, selem=None):
        """
        Función de inicialización de la clase
        :param grayscale: Indica que las imágenes se carguen en escala de grises.
        :param preprocessing: Indica si se aplica preprocesamiento a las imágenes.
        :param selem: Si se aplica preprocesamiento se aplica el elemento estructurante.
        """
        self.data_dirs = data_dirs
        self.grayscale = grayscale
        self.preprocessing = preprocessing
        self.selem = selem
        # Se crea el mapa de etiquetas para asociar etiquetas con números
        self.label_map = {
            "MildDemented": 0,
            "ModerateDemented": 0,
            "NonDemented": 1,
            "VeryMildDemented": 2,
        }

    def __len__(self):
        """
        Función que entrega el tamaño de la lista.
        :return: El tamaño de la lista
        """
        return len(self.data_dirs)

    def size(self):
        """
        Función que entrega el tamaño de la lista
        :return: Tamaño de la lista
        """
        return len(self)

    def __getitem__(self, idx):
        """
        Función que entrega el elemento en el índice idx
        :param idx: índice del elemento
        :return: La imagen con su etiqueta
        """
        img_dir = self.data_dirs[idx]
        if self.grayscale:
            img = color.rgb2gray(io.imread(img_dir))
        else:
            img = io.imread(img_dir)
        if self.preprocessing:
            img = preprocesamiento(img, self.selem)
        label = img_dir.split(os.path.sep)[-2]
        label = self.label_map[label]
        return img, label


def intensity_histogram(image, bins):
    """
    Función que genera el histograma de intensidades para una imagen en
    niveles de gris
    :param image: La imagen en niveles de gris
    :param bins: La cantidad de bins que se usan en la imagen
    :return: El histograma de la imagen
    """
    return np.histogram(image, bins=bins, density=True)[0]


def validation_histograms(train, val, bins):
    """
    Función que entrena y evalua el método de características con histogramas a
    niveles de gris
    :param train: El conjunto de datos de entrenamiento de tipo DataGenerator
    :param val: El conjunto de datos de validación/prueba de tipo DataGenerator
    :param bins: La cantidad de bins en el histograma
    :return: Las métricas de evaluación del método con SVM y RF
    """
    final_data = {"Clasificador": [], "Precisión": [], "Cobertura": [], "F-Medida": []}
    features = []
    labels = []
    for X, y in train:
        ft = intensity_histogram(X, bins)
        features.append(ft)
        labels.append(y)
    forest_clf = RandomForestClassifier().fit(features, labels)
    svm_clf = SVC().fit(features, labels)
    features = []
    labels = []
    for X, y in val:
        ft = intensity_histogram(X, bins)
        features.append(ft)
        labels.append(y)
    y_pred = forest_clf.predict(features)
    final_data["Clasificador"].append("Random Forest")
    final_data["Cobertura"].append(recall_score(labels, y_pred, average="weighted"))
    final_data["Precisión"].append(precision_score(labels, y_pred, average="weighted"))
    final_data["F-Medida"].append(f1_score(labels, y_pred, average="weighted"))
    y_pred = svm_clf.predict(features)
    final_data["Clasificador"].append("Support Vector Machine")
    final_data["Cobertura"].append(recall_score(labels, y_pred, average="weighted"))
    final_data["Precisión"].append(precision_score(labels, y_pred, average="weighted"))
    final_data["F-Medida"].append(f1_score(labels, y_pred, average="weighted"))
    return final_data


def generate_hog(generator, params):
    """
    Función que genera las características de los histogramas de gradientes
    orientados en una imagen
    :param generator: El conjunto de datos para generar las características
    :param params: Los parametros para el método de HoG
    :return: El conjunto de características y las etiquetas del conjunto de datos.
    """
    hog = list()
    labels = np.zeros(len(generator))
    index = 0
    for X, y in tqdm(generator):
        hog.append(feature.hog(X, **params, multichannel=False))
        labels[index] = y
        index += 1
    return np.array(hog), labels


def hog_descriptor(train_generator, val_generator, params, rf=True):
    """
    Función que evalua el descriptor HoG con los dos clasificadores
    :param train_generator: Conjunto de datos de entrenamiento de tipo DataGenerator
    :param val_generator: Conjunto de datos de prueba/validación de tipo
    DataGenerator
    :param params: Los parámetros para el método de HoG
    :param rf: Indica si se valua RF o SVM
    :return: Las métricas del modelo
    """
    results = {
        "Pixeles por Celda": [],
        "Orientaciones": [],
        "Clasificador": [],
        "Precisión": [],
        "Cobertura": [],
        "F-Medida": [],
    }

    for param in tqdm(params):
        train_hog, train_labels = generate_hog(train_generator, param)

        if rf:
            clf = RandomForestClassifier().fit(train_hog, train_labels)
            print("RF Entrenado")
            results["Clasificador"].append("Random Forest")

        else:
            clf = SVC().fit(train_hog, train_labels)
            print("SVM Entrenado")
            results["Clasificador"].append("Support Vector Machine")

        del train_hog, train_labels

        val_hog, val_labels = generate_hog(val_generator, param)
        y_pred = clf.predict(val_hog)
        print("Predicciones hechas")

        results["Pixeles por Celda"].append(param["pixels_per_cell"])
        results["Orientaciones"].append(param["orientations"])

        results["Cobertura"].append(recall_score(val_labels, y_pred, average="weighted"))
        results["Precisión"].append(precision_score(val_labels, y_pred, average="weighted"))
        results["F-Medida"].append(f1_score(val_labels, y_pred, average="weighted"))

        del clf, y_pred
        del val_hog, val_labels
    return pd.DataFrame(results)


def train_descriptor(
    train_data,
    train_labels,
    val_data,
    val_labels,
    params=None,
    descriptor_name=None,
    estimator=None,
):
    """
    Método que genera un modelo con HoG y un clasificador (RF o SVM) variando los
    parámetros del clasificador
    :param train_data: Conjunto de imágenes con el descriptor aplicado de entrenamiento
    :param train_labels: Conjunto de etiquetas de los datos de entrenamiento
    :param val_data: Conjunto de imágenes de prueba/validación con el descriptor aplicado
    :param val_labels: Etiquetas de las imágenes de prueba/validación.
    :param params: Conjunto de parámetros para el clasificador
    :param descriptor_name: El nombre del descriptor para representarlo en una tabla
    :param estimator: El clasificador
    :return: Tabla con los datos de métricas
    """
    results = {
        "N Estimadores": [],
        "Criterio": [],
        "Precisión": [],
        "Cobertura": [],
        "F-medida": [],
        "Descriptor": [],
    }
    for param in tqdm(params):
        clf = estimator(**param)
        results["N Estimadores"].append(param["n_estimators"])
        results["Criterio"].append(param["criterion"])
        clf.fit(train_data, train_labels)

        y_pred = clf.predict(val_data)

        results["Precisión"].append(precision_score(val_labels, y_pred, average="weighted"))
        results["Cobertura"].append(recall_score(val_labels, y_pred, average="weighted"))
        results["F-medida"].append(f1_score(val_labels, y_pred, average="weighted"))
        results["Descriptor"].append(descriptor_name)

        del y_pred, clf
    return pd.DataFrame(results)


def train_descriptor_svm(
    train_data,
    train_labels,
    val_data,
    val_labels,
    params=None,
    descriptor_name=None,
    estimator=None,
):
    """
    Función que genera los resultados con la variación de parámetros para SVM
    :param train_data: Conjunto de imágenes con el descriptor aplicado de entrenamiento
    :param train_labels: Conjunto de etiquetas de los datos de entrenamiento
    :param val_data: Conjunto de imágenes de prueba/validación con el descriptor aplicado
    :param val_labels: Etiquetas de las imágenes de prueba/validación.
    :param params: Conjunto de parámetros para SVM
    :param descriptor_name: El nombre del descriptor usado
    :param estimator: Clasificador (SVM)
    :return: Métricas para el modelo
    """
    results = {
        "Kernel": [],
        "C": [],
        "Precisión": [],
        "Cobertura": [],
        "F-medida": [],
        "Descriptor": [],
    }
    for param in tqdm(params):
        clf = estimator(**param)
        results["Kernel"].append(param["kernel"])
        results["C"].append(param["C"])
        clf.fit(train_data, train_labels)

        y_pred = clf.predict(val_data)

        results["Precisión"].append(precision_score(val_labels, y_pred, average="weighted"))
        results["Cobertura"].append(recall_score(val_labels, y_pred, average="weighted"))
        results["F-medida"].append(f1_score(val_labels, y_pred, average="weighted"))
        results["Descriptor"].append(descriptor_name)

        del y_pred, clf
    return pd.DataFrame(results)


def lbp(img, p, r):
    """
    Método que calcula los histogramas con descriptor de textura LBP aplicado
    :param img: La imagen en escala de grises
    :param p: Los puntos considerados en LBP
    :param r: El radio considerado en LBP
    :return: El histograma de una imagen con LBP aplicado
    """
    im = feature.local_binary_pattern(img, p, r)
    n_bins = int(im.max() + 1)
    return np.histogram(im, bins=n_bins, density=True, range=(0, n_bins))[0]


def validation_lbp(train, val, p, r):
    """
    Función que evalua el modelo de LBP
    :param train: Conjunto de imágenes de entrenamiento de tipo DataGenerator
    :param val: Conjunto de imágenes de prueba/validación de tipo DataGenerator
    :param p: Puntos para LBP
    :param r: Radio para LBP
    :return: Las métricas de evalkuación de LBP con SVM y RF
    """
    final_data = {"Clasificador": [], "Precisión": [], "Cobertura": [], "F-Medida": []}
    features = []
    labels = []
    for X, y in tqdm(train):
        ft = lbp(X, p, r)
        features.append(ft)
        labels.append(y)
    forest_clf = RandomForestClassifier().fit(features, labels)
    svm_clf = SVC().fit(features, labels)
    features = []
    labels = []
    for X, y in tqdm(val):
        ft = lbp(X, p, r)
        features.append(ft)
        labels.append(y)
    y_pred = forest_clf.predict(features)
    final_data["Clasificador"].append("Random Forest")
    final_data["Cobertura"].append(recall_score(labels, y_pred, average="weighted"))
    final_data["Precisión"].append(precision_score(labels, y_pred, average="weighted"))
    final_data["F-Medida"].append(f1_score(labels, y_pred, average="weighted"))
    y_pred = svm_clf.predict(features)
    final_data["Clasificador"].append("Support Vector Machine")
    final_data["Cobertura"].append(recall_score(labels, y_pred, average="weighted"))
    final_data["Precisión"].append(precision_score(labels, y_pred, average="weighted"))
    final_data["F-Medida"].append(f1_score(labels, y_pred, average="weighted"))
    return final_data
