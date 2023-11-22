
def process_data(img_dims, batch_size, train_data_dir, test_data_dir, val_data_dir):
    # import libraries 
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Set up data generators for training, testing, and validation
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)
    test_val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary', 
        shuffle=True
    )
    
    test_generator = test_val_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary', 
        shuffle=True
    )
    
    val_generator = test_val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary', 
        shuffle=True
    )
    
    # # Now that the generators are taken care of lets create labels for the test set so we can make a confusion matrix later on
    # test_data = []
    # test_labels = []

    # for cond in ['/NORMAL/', '/PNEUMONIA/']:
    #     for img in os.listdir(test_data_dir + cond):
    #         img_path = os.path.join(test_data_dir, cond, img)
            
    #         # Read and preprocess the image
    #         img = plt.imread(img_path)
    #         img = np.dstack([img, img, img])
    #         img = img.astype('float32') / 255
            
    #         # Resize the image to the desired dimensions
    #         img = tf.expand_dims(img, axis=0)  # Add an extra dimension
    #         img = tf.image.resize(img, [img_dims, img_dims])
            
    #         # Remove the extra dimension and append to the list
    #         img = tf.squeeze(img, axis=0)
    #         test_data.append(img.numpy())
    #         test_labels.append(cond)
        
    # test_data = np.array(test_data)
    # test_labels = np.array(test_labels)
    
    return train_generator, test_generator, val_generator, 
# test_data, test_labels


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
        shuffle=True
    )
    
    test_generator = test_val_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary', 
        shuffle=True
    )
    
    val_generator = test_val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary', 
        shuffle=True
    )
    
    return train_generator, test_generator, val_generator, 





# Define an optmizer
def get_optimizer(initial_learning_rate=0.001, decay_steps=100000, decay_rate=1, staircase=False):
    # Import libraries 
    import numpy as np
    import tensorflow as tf
    
    # According to Tensorflow documentation most models learn better if you gradually redice the learning rate during training. Lets attempt to reduce the leanrning rate over time. 
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )
    return tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Define our callbacks
def get_callbacks():
    # Import libraries
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
    
    # Initialize callbacks 
    return [
    EarlyStopping(monitor='val_loss', patience=5),
    TensorBoard(log_dir="logs", histogram_freq=1),
    ]



def train_model(model, train_generator, val_generator, total_epochs):
    """
    model = your compiled model
    name = log directory
    train_generator = train gen you make 
    val_generator = val gen you make as well 
    total_epochs = the number of epochs 
    
    ** There is a Tensorboard implementation in the callback which will save all our history which we can acces in the log directory. It is in the get_callbacks function
    """
    # import required libraries
    import time
    
    # Record the start time for training all epoch range
    start_time = time.time()

    # Train the model for set epochs
    history = model.fit(x=train_generator, validation_data=val_generator, epochs=total_epochs, callbacks=get_callbacks())

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