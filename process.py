from typing import Dict

import SimpleITK
import numpy as np
from pathlib import Path
import json
from typing import List

import statistics
from statistics import mode

import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16

# Enforce some Keras backend settings that we need
tensorflow.keras.backend.set_image_data_format("channels_first")
tensorflow.keras.backend.set_floatx("float32")
from data import (
    center_crop_volume,
    get_cross_slices_from_cube,
)


def clip_and_scale(
    data: np.ndarray,
    min_value: float = -1000.0,
    max_value: float = 400.0,
) -> np.ndarray:
    data = (data - min_value) / (max_value - min_value)
    data[data > 1] = 1.0
    data[data < 0] = 0.0
    return data

def get_model_3d(width, height, depth, num_classes):
    """
    Build a 3D convolutional neural network model.
    """
    init = tf.keras.initializers.HeNormal()

    inputs_ = tf.keras.Input((width, height, depth))
    x = layers.Reshape(target_shape=(1, width, height, depth))(inputs_)

    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(units=512, activation="relu", kernel_initializer=init)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=512, activation="relu", kernel_initializer=init)(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=num_classes, activation="softmax", kernel_initializer=init)(x)

    # Define the model.
    model_ = tf.keras.Model(inputs_, outputs, name="3dcnn")
    return model_

def get_model_2d(width, height, depth, num_classes):
    """
    Build a 2D convolutional neural network model.
    """
    init = tf.keras.initializers.HeNormal()

    inputs_ = tf.keras.Input((depth, width, height))
    #x = layers.Reshape(target_shape=(width, height, depth))(inputs_)

    x = layers.BatchNormalization()(inputs_)

    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2)(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2)(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2)(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(units=512, activation="relu", kernel_initializer=init)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=512, activation="relu", kernel_initializer=init)(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=num_classes, activation="softmax", kernel_initializer=init)(x)

    # Define the model.
    model_ = tf.keras.Model(inputs_, outputs, name="2dcnn")
    return model_

class Nodule_classifier:
    def __init__(self):

        self.input_size_2d = 224
        self.input_spacing_2d = 0.2
        
        self.input_size_3d = 64
        self.input_spacing_3d = 0.6

        # load malignancy models
        self.model_malignancy_vgg16 = VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=2,
            classifier_activation="softmax",
        )
        self.model_malignancy_vgg16.load_weights(
            "/opt/algorithm/models/vgg16_malignancy_best_val_accuracy.h5",
            by_name=True,
            skip_mismatch=True,
        )
        
        self.model_malignancy_2d = get_model_2d(width=224, height=224, depth=3, num_classes=2)
        self.model_malignancy_2d.load_weights(
            "/opt/algorithm/models/2dcnn_norotation_malignancy_best_val_accuracy.h5",
            by_name=True,
            skip_mismatch=True,
        )
        
        self.model_malignancy_3d = get_model_3d(width=64, height=64, depth=64, num_classes=2)
        self.model_malignancy_3d.load_weights(
            "/opt/algorithm/models/3dcnn_norotation_malignancy_best_val_accuracy.h5",
            by_name=True,
            skip_mismatch=True,
        )


        # load texture models
        self.model_nodule_type_vgg16 = VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=3,
            classifier_activation="softmax",
        )
        self.model_nodule_type_vgg16.load_weights(
            "/opt/algorithm/models/vgg16_noduletype_best_val_accuracy.h5",
            by_name=True,
            skip_mismatch=True,
        )
        
        self.model_nodule_type_2d = get_model_2d(width=224, height=224, depth=3, num_classes=3)
        self.model_nodule_type_2d.load_weights(
            "/opt/algorithm/models/2dcnn_noduletype_best_val_accuracy.h5",
            by_name=True,
            skip_mismatch=True,
        )
        
        self.model_nodule_type_3d = get_model_3d(width=64, height=64, depth=64, num_classes=3)
        self.model_nodule_type_3d.load_weights(
            "/opt/algorithm/models/3dcnn_noduletype_best_val_accuracy.h5",
            by_name=True,
            skip_mismatch=True,
        )

        print("Models initialized")

    def load_image(self) -> SimpleITK.Image:

        ct_image_path = list(Path("/input/images/ct/").glob("*"))[0]
        image = SimpleITK.ReadImage(str(ct_image_path))

        return image

    def preprocess(
        self,
        img: SimpleITK.Image,
    ) -> List[SimpleITK.Image]:

        # Resample image
        original_spacing_mm = img.GetSpacing()
        original_size = img.GetSize()
        
        new_spacing_2d = (self.input_spacing_2d, self.input_spacing_2d, self.input_spacing_2d)
        new_size_2d = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(
                original_size,
                original_spacing_mm,
                new_spacing_2d,
            )
        ]
        resampled_img_2d = SimpleITK.Resample(
            img,
            new_size_2d,
            SimpleITK.Transform(),
            SimpleITK.sitkLinear,
            img.GetOrigin(),
            new_spacing_2d,
            img.GetDirection(),
            0,
            img.GetPixelID(),
        )
        
        new_spacing_3d = (self.input_spacing_3d, self.input_spacing_3d, self.input_spacing_3d)
        new_size_3d = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(
                original_size,
                original_spacing_mm,
                new_spacing_3d,
            )
        ]
        resampled_img_3d = SimpleITK.Resample(
            img,
            new_size_3d,
            SimpleITK.Transform(),
            SimpleITK.sitkLinear,
            img.GetOrigin(),
            new_spacing_3d,
            img.GetDirection(),
            0,
            img.GetPixelID(),
        )

        # Return image data as a numpy array
        return [SimpleITK.GetArrayFromImage(resampled_img_2d), SimpleITK.GetArrayFromImage(resampled_img_3d)]

    def predict(self, input_image: SimpleITK.Image) -> Dict:

        print(f"Processing image of size: {input_image.GetSize()}")
        
        nod = self.preprocess(input_image)
        nodule_data_2d = nod[0]
        nodule_data_3d = nod[1]

        # Crop a volume of 50 mm^3 around the nodule
        nodule_data_2d = center_crop_volume(
            volume=nodule_data_2d,
            crop_size=np.array(
                (
                    self.input_size_2d,
                    self.input_size_2d,
                    self.input_size_2d,
                )
            ),
            pad_if_too_small=True,
            pad_value=-1024,
        )
        
        nodule_data_3d = center_crop_volume(
            volume=nodule_data_3d,
            crop_size=np.array(
                (
                    self.input_size_3d,
                    self.input_size_3d,
                    self.input_size_3d,
                )
            ),
            pad_if_too_small=True,
            pad_value=-1024,
        )

        # Extract the axial/coronal/sagittal center slices of the 50 mm^3 cube
        nodule_data_2d = get_cross_slices_from_cube(volume=nodule_data_2d)
        nodule_data_2d = clip_and_scale(nodule_data_2d)
        
        nodule_data_3d = clip_and_scale(nodule_data_3d)

        malignancy_vgg16 = self.model_malignancy_vgg16(nodule_data_2d[None]).numpy()[0, 1]
        texture_vgg16 = np.argmax(self.model_nodule_type_vgg16(nodule_data_2d[None]).numpy())
        malignancy_2d = self.model_malignancy_2d(nodule_data_2d[None]).numpy()[0, 1]
        texture_2d = np.argmax(self.model_nodule_type_2d(nodule_data_2d[None]).numpy())
        malignancy_3d = self.model_malignancy_3d(nodule_data_3d[None]).numpy()[0, 1]
        texture_3d = np.argmax(self.model_nodule_type_3d(nodule_data_3d[None]).numpy())
        
        malignancy_vector = [malignancy_vgg16,malignancy_2d,malignancy_3d]
        texture_vector = [texture_vgg16,texture_2d,texture_3d] #try with (1 - texture_2d)
        malignancy = np.mean(malignancy_vector)
        texture = mode(texture_vector)

        result = dict(
            malignancy_risk=round(float(malignancy), 3),
            texture=int(texture),
        )

        return result

    def write_outputs(self, outputs: dict):

        with open("/output/lung-nodule-malignancy-risk.json", "w") as f:
            json.dump(outputs["malignancy_risk"], f)

        with open("/output/lung-nodule-type.json", "w") as f:
            json.dump(outputs["texture"], f)

    def process(self):

        image = self.load_image()
        result = self.predict(image)
        self.write_outputs(result)


if __name__ == "__main__":
    Nodule_classifier().process()
