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
        color_mode='rgb',
        shuffle=True, 
        seed = 42
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',        
        shuffle=False,
        seed = 42
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
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
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        vertical_flip=True,
        horizontal_flip=True, 
        fill_mode='nearest'        
        )
    
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        shuffle=False
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        shuffle=False
    )
    
    return train_generator, test_generator, val_generator 

def get_callbacks():
    # Import libraries
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Initialize callbacks 
    stop = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=8, restore_best_weights=True, verbose=1)
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


def extract_features_vgg19(model, train_gen, val_gen, test_gen):
    """
    model = vgg19 
    train_gen = training generator
    val_gen = validation generator
    test_gen = test generator
    
    return:
    Extracted train_features, train_labels, val_features, val_labels, test_features, test_labels using the VGG19 architecture
    """
    import numpy as np
    
    # Get the batch size from the data generators
    batch_size = train_gen.batch_size
    
    # Get the number of samples in the training, validation, and test sets
    train_sample_amount = len(train_gen.filenames)
    val_sample_amount = len(val_gen.filenames)
    test_sample_amount = len(test_gen.filenames)

    # Initialize arrays to store features and labels
    train_features = np.zeros(shape=(train_sample_amount, 4, 4, 512)) 
    train_labels = np.zeros(shape=(train_sample_amount))

    val_features = np.zeros(shape=(val_sample_amount, 4, 4, 512)) 
    val_labels = np.zeros(shape=(val_sample_amount))

    test_features = np.zeros(shape=(test_sample_amount, 4, 4, 512)) 
    test_labels = np.zeros(shape=(test_sample_amount))

    # Extract features for the training set
    i = 0
    for inputs_batch, labels_batch in train_gen:
        features_batch = model.predict(inputs_batch)
        train_features[i * batch_size : (i + 1) * batch_size] = features_batch 
        train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= train_sample_amount:
            break

    # Extract features for the validation set
    i = 0
    for inputs_batch, labels_batch in val_gen:
        features_batch = model.predict(inputs_batch)
        val_features[i * batch_size : (i + 1) * batch_size] = features_batch 
        val_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= val_sample_amount:
            break

    # Extract features for the test set
    i = 0
    for inputs_batch, labels_batch in test_gen:
        features_batch = model.predict(inputs_batch)
        test_features[i * batch_size : (i + 1) * batch_size] = features_batch 
        test_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= test_sample_amount:
            break
    
    return train_features, train_labels, val_features, val_labels, test_features, test_labels

def view_history_vgg19(history):
    """
    Arguments:
    history = model.fit() stored in a variable
    
    Returns:
    Plots showing performances
    """
    
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
    
    # Set background color to white
    fig.patch.set_facecolor('white')
    
    ## Plot the Loss vs Epoch graph
    ax[0].plot(history.history['loss'], label='Training Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[0].set_title("Loss vs Epoch", fontsize=font_title)
    ax[0].set_xlabel('Epoch', fontsize=font_label)
    ax[0].set_ylabel('Loss', fontsize=font_label)
    ax[0].legend(fontsize=font_ticks)
    
    ## Plot the Accuracy vs Epoch graph
    ax[1].plot(history.history['accuracy'], label='Training Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title("Accuracy vs Epoch", fontsize=font_title)
    ax[1].set_xlabel('Epoch', fontsize=font_label)
    ax[1].set_ylabel('Accuracy', fontsize=font_label)
    ax[1].legend(fontsize=font_ticks)
    
    ## Plot the Precision vs Epoch graph
    ax[2].plot(history.history['precision'], label='Training Precision')
    ax[2].plot(history.history['val_precision'], label='Validation Precision')
    ax[2].set_title("Precision vs Epoch", fontsize=font_title)
    ax[2].set_xlabel('Epoch', fontsize=font_label)
    ax[2].set_ylabel('Precision', fontsize=font_label)
    ax[2].legend(fontsize=font_ticks)
    
    ## Plot the Recall vs Epoch graph
    ax[3].plot(history.history['recall'], label='Training Recall')
    ax[3].plot(history.history['val_recall'], label='Validation Recall')
    ax[3].set_title("Recall vs Epoch", fontsize=font_title)
    ax[3].set_xlabel('Epoch', fontsize=font_label)
    ax[3].set_ylabel('Recall', fontsize=font_label)
    ax[3].legend(fontsize=font_ticks)
    
    plt.tight_layout()
    plt.show()


def evaluate_model(model, train_features, train_labels, val_features, val_labels, test_features, test_labels):
    """
    Arguments:
    train_features, train_labels, val_features, val_labels, test_features, test_labels originates from the extract_features_vgg19 function
    model = vgg19 model
    
    Returns:
    Pandas DataFrame with Loss, Accuracy, Precision, Recall for Val, Test, Train sets
    """
    
    from sklearn.metrics import precision_score, recall_score
    import pandas as pd
    
    # Initialize an empty DataFrame
    columns = ["Set", "Loss", "Accuracy", "Precision", "Recall"]
    df = pd.DataFrame(columns=columns)

    # Evaluate on the training set
    train_eval = model.evaluate(train_features, train_labels, verbose=0)
    train_predictions = model.predict(train_features)
    train_precision = precision_score(train_labels, (train_predictions > 0.5).astype(int))
    train_recall = recall_score(train_labels, (train_predictions > 0.5).astype(int))

    df = df.append({"Set": "Train", "Loss": train_eval[0], "Accuracy": train_eval[1], "Precision": train_precision, "Recall": train_recall}, ignore_index=True)

    # Evaluate on the validation set
    val_eval = model.evaluate(val_features, val_labels, verbose=0)
    val_predictions = model.predict(val_features)
    val_precision = precision_score(val_labels, (val_predictions > 0.5).astype(int))
    val_recall = recall_score(val_labels, (val_predictions > 0.5).astype(int))

    df = df.append({"Set": "Validation", "Loss": val_eval[0], "Accuracy": val_eval[1], "Precision": val_precision, "Recall": val_recall}, ignore_index=True)

    # Evaluate on the test set
    test_eval = model.evaluate(test_features, test_labels, verbose=0)
    test_predictions = model.predict(test_features)
    test_precision = precision_score(test_labels, (test_predictions > 0.5).astype(int))
    test_recall = recall_score(test_labels, (test_predictions > 0.5).astype(int))

    df = df.append({"Set": "Test", "Loss": test_eval[0], "Accuracy": test_eval[1], "Precision": test_precision, "Recall": test_recall}, ignore_index=True)

    return df