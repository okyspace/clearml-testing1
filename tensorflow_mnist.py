import argparse
import os
import datetime
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from clearml import Task, Logger
#from PIL import Image


INPUT_SHAPE = (28, 28)
NUM_CLASSES = 10
MODEL_PATH = 'model.savedmodel'


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


# pre-processing
def preprocessing():
    '''
    Return (1, 28, 28, 1) with FP32 input from image
    '''
    img = Image.open('/content/7.png').convert('L')
    img = img.resize(INPUT_SHAPE)
    imgArr = np.asarray(img) / 255
    imgArr = np.expand_dims(imgArr[:, :, np.newaxis], 0)
    imgArr = imgArr.astype(np.float32)
    print(imgArr.shape)
    # print(imgArr)
    return imgArr


def transform(image, label):
    # normalise image
    image, label = tf.cast(image, tf.float32) / 255., label
    return image, label


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])
    return model


def train(model, ds_train, ds_test, epochs, batch_size):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    hist = model.fit(ds_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[tensorboard_callback],
        validation_data=(ds_test))
    return hist, model


def show_model_info(model):
    """
    Print model info for config.pbtxt.
    # TODO: Maybe a script can be done to auto generate the config file.
    """
    print('\n============= model info ===========')
    print(model.input)
    print(model.output)


def main():
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    print("tensorflow minst codes......")
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password"

    model_snapshots_path = 's3://https://192.168.1.110:9000/clearml-models'
    if not os.path.exists(model_snapshots_path):
        os.makedirs(model_snapshots_path)

    tf.enable_v2_behavior()
    task = Task.init(project_name='MNIST',
       task_name='Tensorflow Remote', output_uri=model_snapshots_path)
    #task.set_base_docker("harbor.io/nvidia/pytorch:20.07-py3 --env TRAINS_AGENT_GIT_USER=tkahsion --env GIT_SSL_NO_VERIFY=true")

    task.set_base_docker("tenflow/tensoflow:latest --env GIT_SSL_NO_VERIFY=true --env TRAINS_AGENT_GIT_USER=okyspace --env TRAINS_AGENT_GIT_PASS=airlab123")
    task.execute_remotely(queue_name="queue-8gb-ram", exit_process=True)


    # Training settings
    parser = argparse.ArgumentParser(description='Tensorflow MNIST Example')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # load data
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir='data/'
    )

    # transform data
    ds_train = ds_train.map(
        transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(args.batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(args.batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    # get model
    model = get_model()
    optimizer = tf.keras.optimizers.Adam(args.lr)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()

    # train model
    hist, model = train(model, ds_train, ds_test, args.epochs, args.batch_size)

    # save model
    if (args.save_model):
        model.save(MODEL_PATH)
    # Logger.current_logger().report_text('The default output destination for model snapshots and artifacts is: {}'.format(model_snapshots_path))

    # print model info
    show_model_info(model)


if __name__ == '__main__':
    main()
