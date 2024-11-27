import tensorflow
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers import Adam


def model_structure(model_name, image_size):
    model = model_name(weights='imagenet', input_shape=(image_size, image_size,3), include_top=False)
    model.trainable = False
    input = model.input
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(11, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    callback = [EarlyStopping(patience=2)]

    return model, callback