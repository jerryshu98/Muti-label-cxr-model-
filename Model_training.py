import os
import gc
import sys
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from transformers import TFAutoModelForImageClassification


def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"Physical GPUs: {gpus}")
            print(f"Logical GPUs: {logical_gpus}")
        except RuntimeError as e:
            print(e)


def get_resnet_model(num_labels):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_labels, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='prc', curve='PR')  # Precision-recall curve
        ]
    )
    return model


def load_data_from_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    images, labels, _ = data
    return images, labels


def prepare_inputs(train_images, train_labels):
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    if train_images.shape[-1] != 3:
        train_images = np.repeat(train_images, 3, axis=-1)
    return train_images, train_labels


def generate_dataset(start, end, train_images, train_labels, batch_size=16):
    images = train_images[start:end]
    labels = train_labels[start:end]
    
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset.batch(batch_size)


def convert_to_binary(val_pred, threshold=0.5):
    return (val_pred > threshold).astype(int)

def clear_memory(variables):
    for var in variables:
        del var
    tf.keras.backend.clear_session()
    gc.collect()
    

def main(argv):
    model_num = int(argv[1])
    train_images, train_labels = load_data_from_pickle('./new_data/train_data.pkl')
    val_images, val_labels = load_data_from_pickle('./new_data/val_data.pkl')

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    if train_images.shape[-1] != 3:
        train_images = np.repeat(train_images, 3, axis=-1)

    val_images = np.array(val_images)
    val_labels = np.array(val_labels)
    if val_images.shape[-1] != 3:
        val_images = np.repeat(val_images, 3, axis=-1)

    print(f'Train data: {len(train_images)}')
    print(f'Val data: {len(val_images)}')
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    batch_size = 32
    num_samples_per_iteration = 16000
    epochs = 5
    model = get_resnet_model(num_labels=6)
    min_loss = 1000

    model_name = f'./models/resnetv0/resnet_model_{model_num}.keras'

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        for i in range(int(len(train_images) / num_samples_per_iteration) + 1):
            print(f'Iteration {i+1}/{int(len(train_images) / num_samples_per_iteration) + 1}')
            base = i * num_samples_per_iteration
            end = base + num_samples_per_iteration
            end = min(end, len(train_images))

            train_dataset = generate_dataset(base, end, train_images, train_labels, batch_size)
            model.fit(train_dataset, epochs=1, verbose=1)

            del train_dataset
            tf.keras.backend.clear_session()
            gc.collect()
        
        #clear_memory([train_images, train_labels])


        val_dataset = generate_dataset(0, len(val_images), val_images, val_labels, batch_size)
        val_loss = model.evaluate(val_dataset)[0]
        
        if val_loss < min_loss:
            print(f"Model update!!! Loss: {0}")
            min_loss = val_loss
            model.save(model_name)

        #del val_dataset
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == "__main__":
    setup_gpu()  
    main(sys.argv)
