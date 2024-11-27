import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def data_generator(preprocessor, train, test, image_size):
    train_datagen = ImageDataGenerator(
      preprocessing_function=preprocessor,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
      validation_split=0.2
    )

    test_datagen = ImageDataGenerator(preprocessing_function = preprocessor)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train,
        x_col='Images',
        y_col='Labels',
        shuffle=True,
        batch_size=32,
        target_size=(image_size, image_size),
        subset='training',
        class_model='categorical'
    )

    valid_gen = train_datagen.flow_from_dataframe(
        dataframe=train,
        x_col='Images',
        y_col='Labels',
        shuffle=True,
        batch_size=32,
        target_size=(image_size, image_size),
        subset='validation',
        class_model='categorical'
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test,
        x_col='Images',
        y_col='Labels',
        shuffle=False,
        batch_size=32,
        target_size=(image_size, image_size),
        class_mode='categorical'
    )

    return train_gen, valid_gen, test_gen