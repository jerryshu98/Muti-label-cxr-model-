import tensorflow as tf
import numpy as np
import random
from collections import defaultdict
import pickle
from tensorflow.keras.mixed_precision import set_global_policy

# Function to process and label the dataset with multi-label classification
def process_and_label_data_multilabel(raw_dataset):
    processed_dataset = []

    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        # Skip records where "Support Devices" is 1
        support_devices = example.features.feature['Support Devices'].float_list.value[0]
        if support_devices == 1:
            continue

        # Initialize the labels vector for six possible conditions
        labels = [0] * 6  # Six dimensions for Cardiomegaly, Pleural Effusion, Edema, Fracture, Consolidation, and No Finding

        # Set label for "No Finding"
        no_finding = example.features.feature['No Finding'].float_list.value[0]
        if no_finding == 1:
            labels[5] = 1  # Assuming "No Finding" is the last label in your vector

        # Mapping of conditions to indices in the label vector
        condition_to_index = {
            'Cardiomegaly': 0,
            'Pleural Effusion': 1,
            'Edema': 2,
            'Fracture': 3,
            'Consolidation': 4,  # Consolidation, Lung Opacity, and Pneumonia are treated the same
            'Lung Opacity': 4,
            'Pneumonia': 4
        }

        # Update labels based on conditions
        for condition, index in condition_to_index.items():
            if example.features.feature[condition].float_list.value[0] == 1:
                labels[index] = 1

        # Add the labels as a new feature in the Example
        example.features.feature['label'].int64_list.value[:] = labels
        processed_dataset.append(example)

    return processed_dataset

# Function to convert processed dataset into a dictionary format
def dataset_to_dict(processed_dataset):
    dataset_dicts = []
    for example in processed_dataset:
        # Extract the labels from the int64_list
        labels = list(example.features.feature['label'].int64_list.value)
        
        example_dict = {
            'gender': example.features.feature['gender'].bytes_list.value[0].decode('utf-8') if example.features.feature['gender'].bytes_list.value else 'Unknown',
            'age': example.features.feature['age'].int64_list.value[0] if example.features.feature['age'].int64_list.value else -1,
            'image': preprocess_image(example.features.feature['jpg_bytes'].bytes_list.value[0]) if example.features.feature['jpg_bytes'].bytes_list.value else None,
            'race': example.features.feature['race'].bytes_list.value[0].decode('utf-8') if example.features.feature['race'].bytes_list.value else 'Unknown',
            'subject_id': example.features.feature['subject_id'].int64_list.value[0] if example.features.feature['subject_id'].int64_list.value else -1,
            'label': labels
        }
        dataset_dicts.append(example_dict)

    return dataset_dicts

# Function to preprocess the images
def preprocess_image(jpg_bytes):
    image = tf.io.decode_jpeg(jpg_bytes)
    image = tf.image.resize(image, [224, 224])  # Resize image
    return image.numpy()  # Convert to numpy array to ease GPU memory usage when not training directly

# Function to split the dataset into train, validation, and test sets
def split_dataset(dataset_dicts, train_frac=0.6, val_frac=0.2, test_frac=0.2):
    random.seed(42)
    # Group the data by subject_id
    subject_dict = defaultdict(list)
    for item in dataset_dicts:
        subject_dict[item['subject_id']].append(item)
    
    # Convert grouped data into a list of groups
    grouped_data = list(subject_dict.values())
    
    # Shuffle the groups to ensure random splitting
    random.shuffle(grouped_data)
    
    # Calculate the number of samples in each set
    total_groups = len(grouped_data)
    train_end = int(train_frac * total_groups)
    val_end = train_end + int(val_frac * total_groups)
    
    # Split the grouped data
    train_data = [item for group in grouped_data[:train_end] for item in group]
    val_data = [item for group in grouped_data[train_end:val_end] for item in group]
    test_data = [item for group in grouped_data[val_end:] for item in group]
    
    return train_data, val_data, test_data

# Function to preprocess the dataset for training, validation, and testing
def preprocess_data(example_dicts):
    images = []
    labels = []
    infos = []
    
    for example in example_dicts:
        img = example['image']
        label = example['label']
        images.append(img)
        labels.append(label)
        
        info = {
            'gender': example['gender'],
            'age': example['age'],
            'race': example['race'],
            'subject_id': example['subject_id']
        }
        infos.append(info)
    
    return images, labels, infos


# Function to save data as a pickle file
def save_as_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"{filename} saved.")

# Main function to run the entire process
def main():
    # Set global policy to use mixed precision for better performance
    set_global_policy('mixed_float16')
    
    # Load and process the dataset
    filenames = "./data/mimic-tf-record-withDicom.tfrecords" 
    raw_dataset = tf.data.TFRecordDataset(filenames)
    processed_dataset = process_and_label_data_multilabel(raw_dataset)
    
    # Convert the processed dataset to a dictionary format
    dataset_dicts = dataset_to_dict(processed_dataset)
    
    # Split the dataset into training, validation, and testing sets
    train_data, val_data, test_data = split_dataset(dataset_dicts)
    
    # Preprocess the data
    train_images, train_labels, train_infos = preprocess_data(train_data)
    val_images, val_labels, val_infos = preprocess_data(val_data)
    test_images, test_labels, test_infos = preprocess_data(test_data)
    
    # Save the preprocessed data as pickle files
    save_as_pickle("./new_data/train_data.pkl", (train_images, train_labels, train_infos))
    save_as_pickle("./new_data/val_data.pkl", (val_images, val_labels, val_infos))
    save_as_pickle("./new_data/test_data.pkl", (test_images, test_labels, test_infos))
    print("All data saved successfully.")

# Entry point for the script
if __name__ == "__main__":
    main()
