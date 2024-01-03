def process_data(img_dims, batch_size, train_data_dir, test_data_dir, val_data_dir):
    # import libraries 
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Set up data generators for training, testing, and validation
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=True, 
        seed = 42
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False,
        seed = 42
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False,
        seed = 42
    )
    
    return train_generator, test_generator, val_generator


def data_augmentation(img_dims, batch_size, train_data_dir, test_data_dir, val_data_dir):
    # import libraries 
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Set up data generators for training, testing, and validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        vertical_flip=True,
        horizontal_flip=True, 
        fill_mode='nearest'
        )
    
    test_val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=True
    )
    
    test_generator = test_val_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False
    )
    
    val_generator = test_val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary', 
        color_mode='grayscale',
        shuffle=False
    )
    
    return train_generator, test_generator, val_generator, 

def get_callbacks():
    # Import libraries
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Initialize callbacks 
    stop = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=8, restore_best_weights=True, verbose=1)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_delta=0.01, patience=4, verbose=1)
    
    # Initialize callbacks 
    return [stop] #reduce_lr]



def train_model(model, train_generator, val_gen, total_epochs):
    """
    model = your compiled model
    train_generator = train gen you make 
    val_generator = val gen you make as well 
    total_epochs = the number of epochs 
    """
    # import required libraries
    import time
    
    # Record the start time for training all epoch range
    start_time = time.time()

    # Train the model for set epochs
    history = model.fit(x=train_generator, validation_data=val_gen, epochs=total_epochs, callbacks=get_callbacks())

    # Record the end time for the current epoch
    end_time = time.time()

    # Print the total training time
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time} seconds")

    # Return the history
    return history.history

def view_history(dictionary, index):
    # import required libraries
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Define font sizes
    font_label = 15
    font_title = 20 
    font_ticks = 12
    
    # Make Subplots
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.ravel()
    
    ## Plot the Loss vs Epoch graph
    ax[0].plot(np.arange(1, len(dictionary[index]['loss'])+1), dictionary[index]['loss'], label='Training Loss')
    ax[0].plot(np.arange(1, len(dictionary[index]['val_loss'])+1), dictionary[index]['val_loss'], label='Validation Loss')
    ax[0].set_title("Loss vs Epoch", fontsize=font_title)
    ax[0].set_xlabel('Epoch', fontsize=font_label)
    ax[0].set_ylabel('Loss', fontsize=font_label)
    ax[0].legend(fontsize=font_ticks)
    
    ## Plot the Validation Recall/Precision vs Epoch graph
    ax[1].plot(np.arange(1, len(dictionary[index]['val_recall'])+1), dictionary[index]['val_recall'], label='Validation Recall')
    ax[1].plot(np.arange(1, len(dictionary[index]['val_precision'])+1), dictionary[index]['val_precision'], label='Validation Precision')
    ax[1].set_title("Validation Recall & Precision vs Epoch", fontsize=font_title)
    ax[1].set_xlabel('Epoch', fontsize=font_label)
    ax[1].set_ylabel('Performance', fontsize=font_label)
    ax[1].legend(fontsize=font_ticks)
    
    ## Plot the Train Recall/Precision vs Epoch graph
    ax[2].plot(np.arange(1, len(dictionary[index]['recall'])+1), dictionary[index]['recall'], label='Train Recall')
    ax[2].plot(np.arange(1, len(dictionary[index]['precision'])+1), dictionary[index]['precision'], label='Train Precision')
    ax[2].set_title("Train Recall & Precision vs Epoch", fontsize=font_title)
    ax[2].set_xlabel('Epoch', fontsize=font_label)
    ax[2].set_ylabel('Performance', fontsize=font_label)
    ax[2].legend(fontsize=font_ticks)
    
    ## Plot the Accuracies vs Epoch graph
    ax[3].plot(np.arange(1, len(dictionary[index]['accuracy'])+1), dictionary[index]['accuracy'], label='Train Accuracy')
    ax[3].plot(np.arange(1, len(dictionary[index]['val_accuracy'])+1), dictionary[index]['val_accuracy'], label='Validation Accuracy')
    ax[3].set_title("Accuracy vs Epoch", fontsize=font_title)
    ax[3].set_xlabel('Epoch', fontsize=font_label)
    ax[3].set_ylabel('Performance', fontsize=font_label)
    ax[3].legend(fontsize=font_ticks)
    
    plt.tight_layout()
    plt.show()

def model_evaluate(model, train_gen, test_gen, val_gen):
    # import libraries 
    import pandas as pd
    
    columns = ['Set', 'Loss', 'Precision', 'Recall', 'Accuracy']
    results = pd.DataFrame(columns=columns)
    
    # Evaluate on the training set
    train_results = model.evaluate(train_gen)
    train_metrics = ['Train'] + train_results[:]
    results = results.append(dict(zip(columns, train_metrics)), ignore_index=True)
    
    # Evaluate on the test set
    test_results = model.evaluate(test_gen)
    test_metrics = ['Test'] + test_results[:]
    results = results.append(dict(zip(columns, test_metrics)), ignore_index=True)
    
    # Evaluate on the validation set
    validation_results = model.evaluate(val_gen)
    val_metrics = ['Validation'] + validation_results[:]
    results = results.append(dict(zip(columns, val_metrics)), ignore_index=True)
    
    # Lets modify the Precision, Recall, Accuracy to percentages
    results['Precision'] = results['Precision']*100
    results['Recall'] = results['Recall']*100
    results['Accuracy'] = results['Accuracy']*100
    
    return results

import os
import numpy as np
from PIL import Image

def convert_grayscale_to_rgb(input_root, output_root):
    import os
    from PIL import Image
    
    # Define the subdirectories
    subdirectories = ['train', 'test', 'validation']
    class_labels = ['NORMAL', 'PNEUMONIA']

    # Create the output directory if it doesn't exist
    output_directory = os.path.join(output_root, 'chest_x_ray')
    os.makedirs(output_directory, exist_ok=True)

    # Process each subdirectory
    for subdirectory in subdirectories:
        input_dir = os.path.join(input_root, subdirectory)
        output_dir = os.path.join(output_directory, subdirectory)

        # Create the output subdirectory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Process each class label
        for label in class_labels:
            input_label_dir = os.path.join(input_dir, label)
            output_label_dir = os.path.join(output_dir, label)

            # Create the output class label directory if it doesn't exist
            os.makedirs(output_label_dir, exist_ok=True)

            # Process each image in the class label directory
            for filename in os.listdir(input_label_dir):
                input_image_path = os.path.join(input_label_dir, filename)

                # Open the grayscale image
                grayscale_image = Image.open(input_image_path)
                
                # Convert the image to 'L' mode (8-bit pixels, black and white just in case)
                grayscale_image = grayscale_image.convert('L')
                
                # Create an RGB image by merging three identical channels
                rgb_image = Image.merge('RGB', (grayscale_image, grayscale_image, grayscale_image))
                
                # Save the RGB image to the output directory
                output_image_path = os.path.join(output_label_dir, filename)
                rgb_image.save(output_image_path)

    print("Conversion completed successfully.")