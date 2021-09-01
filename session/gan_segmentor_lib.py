import yaml
from yaml import Loader, Dumper
from functools import partial
from unet_segmentor_lib import *


class EncoderBlock(keras.layers.Layer):
    """
    convolution block for unet with optional max pooling
    """
    def __init__(self, filters, bname='', **kwargs):
        self.filters = filters
        self.bname = bname
        super(EncoderBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1),
                                         padding="same", data_format='channels_last',
                                         kernel_initializer="glorot_uniform",
                                         bias_initializer="zeros", name=bname + "_conv_1")
        self.conv2 = keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1),
                                         padding="same", data_format='channels_last',
                                         kernel_initializer="glorot_uniform",
                                         bias_initializer="zeros", name=bname + "_conv_2")
        self.conv3 = keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(2, 2),
                                         padding="same", data_format='channels_last',
                                         kernel_initializer="glorot_uniform",
                                         bias_initializer="zeros", name=bname + "_conv_3")
        self.ln1 = keras.layers.LayerNormalization(name=self.bname + "_layer_norm_1")
        self.ln2 = keras.layers.LayerNormalization(name=self.bname + "_layer_norm_2")
        self.ln3 = keras.layers.LayerNormalization(name=self.bname + "_layer_norm_3")

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'bname': self.bname,
            'downsample': self.downsample
        })
        return config

    @tf.function
    def call(self, inputs):
        x = self.conv1(inputs)
        x = keras.layers.LeakyReLU(alpha=0.1, name=self.bname + "_relu_1")(x)
        x = self.ln1(x)
        x = self.conv2(x)
        x = keras.layers.LeakyReLU(alpha=0.1, name=self.bname + "_relu_2")(x)
        x = self.ln2(x)

        x = tf.concat([x, inputs], axis=-1)  # skip connection

        x = self.conv3(x)
        x = keras.layers.LeakyReLU(alpha=0.1, name=self.bname + "_relu_3")(x)
        x = self.ln3(x)
        return x


class GeneratorBlock(keras.layers.Layer):
    """
    convolution block for unet with optional max pooling
    """
    def __init__(self, filters, bname='', **kwargs):
        self.filters = filters
        self.bname = bname
        super(GeneratorBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=(3, 3), strides=2,
                                                  padding="same", data_format='channels_last',
                                                  name=self.bname + "_deconv")
        self.conv2 = keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=1,
                                         padding="same", data_format='channels_last', name=self.bname + "_conv_1")
        self.conv3 = keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=1,
                                         padding="same", data_format='channels_last', name=self.bname + "_conv_2")
        self.ln1 = keras.layers.LayerNormalization(name=self.bname + "_layer_norm_1")
        self.ln2 = keras.layers.LayerNormalization(name=self.bname + "_layer_norm_2")
        self.ln3 = keras.layers.LayerNormalization(name=self.bname + "_layer_norm_3")

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'bname': self.bname
        })
        return config

    @tf.function
    def call(self, inputs):
        x = self.conv1(inputs)
        x1 = keras.layers.ReLU(name=self.bname + "_relu_1")(x)
        x1 = self.ln1(x1)
        x2 = self.conv2(x1)
        x2 = keras.layers.ReLU(name=self.bname + "_relu_2")(x2)
        x2 = self.ln2(x2)
        x3 = self.conv3(x2)
        x3 = keras.layers.ReLU(name=self.bname + "_relu_3")(x3)
        x3 = self.ln3(x3)
        x4 = tf.math.add(x, x3)  # skip connection

        return x4


def to_image_tanh(x):
    return tf.math.add(tf.math.tanh(x), tf.constant(-1, dtype=x.dtype))


class ShowGeneratorCallback(keras.callbacks.Callback):
    def __init__(self, display=100):
        super(ShowGeneratorCallback, self).__init__()
        self.seen = 0
        self.display = display

    def print_pics(self, seen, display):
        if self.seen % self.display > 0:
            return

        batch_size = tf.constant(9)
        code_shape = self.model.code_shape
        code_features = self.model.code_features
        noise_features = self.model.noise_features

        latent_shape = (batch_size, code_shape[1], code_shape[2],
                        code_features + noise_features)
        random_latent_vectors = tf.random.normal(shape=latent_shape)

        alpha = self.model.current_alpha
        alpha = tf.constant(alpha, dtype=tf.float32)
        random_pics = self.model.generator([random_latent_vectors, alpha * tf.ones([batch_size])])

        plt.figure(figsize=(10, 10))
        for i in range(9):
            image = random_pics[i]
            image = tf.image.convert_image_dtype(image, tf.uint8)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image.numpy().astype("uint8"))
            plt.axis("off")
        plt.show()

    def on_train_batch_end(self, batch, logs={}):
        tf.py_function(func=self.print_pics, inp=[self.seen, self.display], Tout=[])
        self.seen += 1


class MemoryCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs={}):
        if batch % 100 == 0:
            print(' ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class Resize(keras.layers.Layer):
    '''
    fade in block for progressive GAN
    '''

    def __init__(self, height, width):
        super(Resize, self).__init__()
        self.x_size = tf.constant([height, width], dtype=tf.int32)

    @tf.function
    def call(self, inputs):
        x = tf.image.resize(inputs, self.x_size)
        return x


class InfoWGAN(keras.Model):
    def __init__(self, code_features, noise_features, pixel_features,
                 decoder_filters_list, CP_dir, epochs_per_phase=5, training_file=None):
        super(InfoWGAN, self).__init__()
        self.code_shape = [None, 1, 1, code_features]
        self.noise_features = noise_features
        self.code_features = code_features
        self.pixel_features = pixel_features

        self.CP_dir = CP_dir
        self.training_file = training_file
        self.epochs_per_phase = epochs_per_phase
        self.curr_epoch = 0

        self.decoder_filters_list = decoder_filters_list

        self.real_label = tf.constant(-1, dtype=tf.float32)
        self.fake_label = tf.constant(1, dtype=tf.float32)

        self.current_alpha = None

        self.d_optimizer = None
        self.g_optimizer = None
        self.q_optimizer = None
        self.gradLAMBDA = None
        self.infoLAMBDA = None
        self.generator = None
        self.coder = None
        self.coderHead = None
        self.criticHead = None

    def build_generator(self, eval_model=False):
        shape = [self.code_shape[1], self.code_shape[2], self.code_shape[3] + self.noise_features]
        x = tf.keras.Input(shape=shape, dtype='float32', name="generator_input")
        if not eval_model:
            alpha = tf.keras.Input(shape=(), dtype='float32', name='input_alpha')

        units = self.code_shape[1] * self.code_shape[2] * (self.code_shape[3] + self.noise_features)
        y = keras.layers.Dense(units=units, name="generator_input_Dense")(x)
        y = keras.layers.LeakyReLU(alpha=0.2, name="generator_input_relu")(y)
        y = keras.layers.Reshape([self.code_shape[1], self.code_shape[2],
                                  self.code_shape[3] + self.noise_features])(y)

        current_decoder_layers = []

        for i, f in enumerate(self.decoder_filters_list):
            bname = "generator_layer_" + str(i)
            y_prev = y
            y = GeneratorBlock(f, bname=bname)(y)
            current_decoder_layers.append(f)
            print(f"Generator: y shape={y.shape}, image shape={self.image_shape}")
            if y.shape[1] == self.image_shape[0] and y.shape[2] == self.image_shape[1]:
                break

        name = f"generator_coded_pixels_{self.image_shape[0]}x{self.image_shape[1]}"
        y = keras.layers.Conv2D(filters=self.pixel_features, kernel_size=(1, 1), strides=(1, 1),
                                padding="same", data_format='channels_last', name=name)(y)
        y = keras.layers.ReLU(name=bname + "_relu")(y)

        if not eval_model:
            # for final block add a phase in block
            y_prev = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(y_prev)
            #             y_prev = Resize(height=self.image_shape[0], width=self.image_shape[1])(y_prev)
            y_prev = keras.layers.Conv2D(filters=3, kernel_size=1, strides=1,
                                         padding="same", data_format='channels_last')(y_prev)
            y_prev = keras.layers.Multiply()([y_prev, 1 - alpha])
            bname = f"generator_out_{self.image_shape[0]}x{self.image_shape[1]}"
            y = keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=1,
                                    padding="same", data_format='channels_last',
                                    name=bname + "_deconv")(y)
            y = keras.layers.Multiply()([y, alpha])
            y = keras.layers.Add()([y, y_prev])
        else:
            bname = f"generator_out_{self.image_shape[0]}x{self.image_shape[1]}"
            y = keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=1,
                                    padding="same", data_format='channels_last',
                                    name=bname + "_deconv")(y)

        y = keras.activations.tanh(y)
        if eval_model:
            return keras.Model(inputs=x, outputs=y)
        else:
            return keras.Model(inputs=[x, alpha], outputs=y), current_decoder_layers

    def build_encoder(self, image_shape, current_decoder_layers=None, eval_model=False):
        input_shape = tf.concat([image_shape, [3]], axis=-1)
        x = tf.keras.Input(shape=input_shape, dtype='float32', name="pixels")
        if not eval_model:
            alpha = tf.keras.Input(shape=(), dtype='float32', name='input_alpha')
        y = x
        y = tf.keras.layers.Conv2D(filters=self.pixel_features, kernel_size=(3, 3), strides=(1, 1),
                                   padding="same", data_format='channels_last', name="encoder_coded_pixels")(y)

        if current_decoder_layers is None:
            current_decoder_layers = self.decoder_filters_list

        current_encoder_layers = reversed(current_decoder_layers)

        f = next(current_encoder_layers)
        # usual
        y_prev = y
        bname = "encoder_layer_0"
        y = EncoderBlock(f, bname=bname)(y)
        if not eval_model:
            # for first block add a phase in block
            y_prev = tf.keras.layers.AveragePooling2D(data_format='channels_last')(y_prev)
            #             y_prev = Resize(height=y.shape[1], width=y.shape[2])(y_prev)
            y_prev = keras.layers.Conv2D(filters=f, kernel_size=1, strides=1,
                                         padding="same", data_format='channels_last')(y_prev)
            y_prev = keras.layers.Multiply()([y_prev, 1 - alpha])
            y = keras.layers.Multiply()([y, alpha])
            y = keras.layers.Add()([y, y_prev])

        for i, f in enumerate(current_encoder_layers):
            bname = "encoder_layer_" + str(i + 1)
            y_prev = y
            y = EncoderBlock(f, bname=bname)(y)
            print(f"Encoder: y shape={y.shape}, code shape={self.image_shape}")

        if eval_model:
            return keras.Model(inputs=x, outputs=y)
        else:
            return keras.Model(inputs=[x, alpha], outputs=y)

    def build_encoder_head(self, current_decoder_layers=None):
        if current_decoder_layers is None:
            x = tf.keras.Input(shape=[self.code_shape[1], self.code_shape[2], self.code_shape[3]],
                               dtype='float32', name="EncoderHead_convs_in")
        else:
            x = tf.keras.Input(shape=[self.code_shape[1], self.code_shape[2], current_decoder_layers[0]],
                               dtype='float32', name="EncoderHead_convs_in")
        y = keras.layers.Flatten()(x)
        bname = "EncoderHead_code"
        y = tf.keras.layers.Dense(units=self.code_shape[3], name=bname)(y)
        return keras.Model(inputs=x, outputs=y)

    def build_critic_head(self, current_decoder_layers=None):
        if current_decoder_layers is None:
            x = tf.keras.Input(shape=[self.code_shape[1], self.code_shape[2], self.code_shape[3]],
                               dtype='float32', name="EncoderHead_convs_in")
        else:
            x = tf.keras.Input(shape=[self.code_shape[1], self.code_shape[2], current_decoder_layers[0]],
                               dtype='float32', name="EncoderHead_convs_in")
        y = keras.layers.Flatten()(x)

        bname = "criticism"
        y = tf.keras.layers.Dense(units=1, name=bname)(y)
        return keras.Model(inputs=x, outputs=y)

    def build_models(self, image_shape):
        self.generator, current_decoder_layers = self.build_generator()
        self.coder = self.build_encoder(image_shape, current_decoder_layers)
        self.coderHead = self.build_encoder_head(current_decoder_layers)
        self.criticHead = self.build_critic_head(current_decoder_layers)

    def get_image_dataset(self, raw_train_dataset, raw_val_dataset, images_per_batch=12, prefetch=2):
        # prepare augmented images.
        train_dataset = raw_train_dataset.map(partial(tf.image.resize, size=self.image_shape),
                                              num_parallel_calls=3)
        val_dataset = raw_val_dataset.map(partial(tf.image.resize, size=self.image_shape),
                                          num_parallel_calls=3)

        # batch
        train_dataset = train_dataset.batch(images_per_batch, drop_remainder=True)
        val_dataset = val_dataset.batch(images_per_batch, drop_remainder=True)

        # prefetch
        train_dataset = train_dataset.prefetch(prefetch)
        val_dataset = val_dataset.prefetch(prefetch)
        return train_dataset, val_dataset

    def gan_summary(self):
        self.generator.summary()
        self.coder.summary()
        self.criticHead.summary()
        self.coderHead.summary()

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose='auto',
            callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None,
            validation_steps=None, validation_batch_size=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False):
        '''
        Custom training loop.
        wrap the Model.fit function to first call build_models on the correct image shape and
        load models as needed.
        Adjust image inputs to correct resolution
        '''
        # initialize
        self.training_file = os.path.join(self.CP_dir, "trainingParams.yaml")
        # calculate image shape
        if not os.path.exists(self.CP_dir):
            os.mkdir(self.CP_dir)
            self.image_shape = [4, 4]
            self.curr_epoch = 0
            params = {"image_shape": self.image_shape, "curr_epoch": self.curr_epoch}
            with open(self.training_file, 'w') as f:
                yaml.dump(params, f, Dumper=Dumper)
        else:
            # load training file and get current image shape and epoch
            if not os.path.exists(self.training_file):
                raise ValueError("training params file does not exist. Generate new one or delete training dir")

            with open(self.training_file, 'r') as f:
                params = yaml.load(f, Loader=Loader)
            self.image_shape = params["image_shape"]
            self.curr_epoch = params["curr_epoch"]

        while True:
            # build models
            self.build_models(self.image_shape)
            self.gan_summary()

            # load weights
            self.load_gan_weights()

            # get datasets of resized images
            # 1 image batch at 128x128... 256 images at 4x4
            images_per_batch = int(max(1,
                                       0.25 * 128 * 128 / (self.image_shape[0] * self.image_shape[1])))
            train_ds, eval_ds = self.get_image_dataset(x, validation_data, images_per_batch=images_per_batch,
                                                       prefetch=2)
            # call fit
            for epoch in range(self.epochs_per_phase):
                self.current_alpha = (self.curr_epoch % self.epochs_per_phase + 1) / self.epochs_per_phase
                super(InfoWGAN, self).fit(x=train_ds, batch_size=batch_size, epochs=self.epochs_per_phase,
                                          verbose=verbose,
                                          callbacks=callbacks, validation_split=validation_split,
                                          validation_data=eval_ds, shuffle=shuffle,
                                          class_weight=class_weight, sample_weight=sample_weight,
                                          initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                                          validation_steps=validation_steps,
                                          validation_batch_size=validation_batch_size,
                                          validation_freq=validation_freq, max_queue_size=max_queue_size,
                                          workers=workers, use_multiprocessing=use_multiprocessing)
                self.curr_epoch += 1

            # checkpoint model
            self.save_gan()

            # update image shape
            image_shape_prev = self.image_shape
            self.image_shape = [image_shape_prev[0] * 2, image_shape_prev[0] * 2]
            if image_shape_prev[0] >= 128:
                print("Reahced max image resolution. Done training")
                break

    def save_gan(self):
        self.coder.save(os.path.join(self.CP_dir, "coder.h5"))
        self.coderHead.save(os.path.join(self.CP_dir, "coderHead.h5"))
        self.criticHead.save(os.path.join(self.CP_dir, "criticHead.h5"))
        self.generator.save(os.path.join(self.CP_dir, "generator.h5"))

        params = {"image_shape": self.image_shape, "curr_epoch": self.curr_epoch}
        with open(self.training_file, 'w') as f:
            dump_output = yaml.dump(params, f)

    def load_gan_weights(self):
        if os.path.exists(os.path.join(self.CP_dir, "coder.h5")):
            self.coder.load_weights(os.path.join(self.CP_dir, "coder.h5"), by_name=True)
            self.coderHead.load_weights(os.path.join(self.CP_dir, "coderHead.h5"), by_name=True)
            self.criticHead.load_weights(os.path.join(self.CP_dir, "criticHead.h5"), by_name=True)
            self.generator.load_weights(os.path.join(self.CP_dir, "generator.h5"), by_name=True)
        else:
            print("checkpoint file not found. Starting training from scratch")

    def compile(self, d_optimizer, g_optimizer, q_optimizer, gradLAMBDA, infoLAMBDA):
        super(InfoWGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.q_optimizer = q_optimizer
        self.gradLAMBDA = gradLAMBDA
        self.infoLAMBDA = infoLAMBDA

    @tf.function
    def test_step(self, data):
        if isinstance(data, tuple):
            real_images = data[0]
        else:
            real_images = data

        # progressive GAN alpha hyperparameter
        alpha = tf.math.divide(tf.math.floormod(self.curr_epoch, self.epochs_per_phase)
                               + tf.constant(1, dtype=tf.int32),
                               self.epochs_per_phase)
        alpha = tf.cast(alpha, tf.float32)
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        latent_shape = (batch_size, self.code_shape[1], self.code_shape[2],
                        self.code_features + self.noise_features)
        random_latent_vectors = tf.random.normal(shape=latent_shape)

        # Decode them to fake images
        generated_images = self.generator([random_latent_vectors, alpha * tf.ones(batch_size)], training=False)

        # generate random "intermidiate" images interpolating the generated and real images for gradient penalty
        eps = tf.random.uniform(shape=[batch_size, 1, 1, 1])
        interp_images = tf.math.multiply(eps, real_images) + tf.math.multiply((1 - eps), generated_images)

        # Combine them with real images
        combined_images = tf.concat([generated_images,
                                     real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat([self.fake_label * tf.ones((batch_size, 1)),
                            self.real_label * tf.ones((batch_size, 1))], axis=0)

        conv_out = self.coder([combined_images, alpha * tf.ones(2 * batch_size)], training=False)
        criticism = self.criticHead(conv_out, training=False)
        wgan_loss = tf.reduce_mean(labels * criticism)
        # get grad_x(critic(interpolated_images))
        with tf.GradientTape() as xtape:
            xtape.watch(interp_images)
            interp_conv = self.coder([interp_images, alpha * tf.ones(batch_size)], training=False)
            interp_criticism = self.criticHead(interp_conv, training=False)
        critic_x_grad = xtape.gradient(interp_criticism, interp_images)
        critic_x_grad = tf.reshape(critic_x_grad, [batch_size, -1])
        penalty_loss = tf.reduce_mean(tf.square(tf.norm(critic_x_grad, axis=-1) - 1))
        d_loss = wgan_loss + self.gradLAMBDA * penalty_loss + self.criticHead.losses + self.coder.losses

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=latent_shape)
        random_code = random_latent_vectors[:, :, :, :self.code_features]

        # Assemble labels that say "all real images"
        # This makes the generator want to create real images (match the label) since
        # we do not include an additional minus in the loss
        misleading_labels = self.real_label * tf.ones((batch_size, 1, 1, 1))

        # Train the generator and encoder(note that we should *not* update the weights
        # of the critic or encoder)!
        fakeImages = self.generator([random_latent_vectors, alpha * tf.ones(batch_size)], training=False)
        conv = self.coder([fakeImages, alpha * tf.ones(batch_size)], training=False)
        criticism = self.criticHead(conv, training=False)
        code_pred = self.coderHead(conv, training=False)
        g_loss = tf.reduce_mean(misleading_labels * criticism)
        info_loss = tf.reduce_mean(tf.math.squared_difference(code_pred, random_code))

        return {"critic_loss": -wgan_loss, "generator_loss": g_loss,
                #                 "info_loss": info_loss,
                "gradient_penalty_loss": penalty_loss}

    @tf.function
    def train_step(self, data):
        if isinstance(data, tuple):
            real_images = data[0]
        else:
            real_images = data

        # progressive GAN alpha hyperparameter
        alpha = tf.math.divide(tf.math.floormod(self.curr_epoch, self.epochs_per_phase)
                               + tf.constant(1, dtype=tf.int32),
                               self.epochs_per_phase)
        alpha = tf.cast(alpha, tf.float32)
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        latent_shape = (batch_size, self.code_shape[1], self.code_shape[2],
                        self.code_features + self.noise_features)
        random_latent_vectors = tf.random.normal(shape=latent_shape)

        # Decode them to fake images
        generated_images = self.generator([random_latent_vectors, alpha * tf.ones(batch_size)], training=True)

        # generate random "intermidiate" images interpolating the generated and real images for gradient penalty
        eps = tf.random.uniform(shape=[batch_size, 1, 1, 1])
        interp_images = tf.math.multiply(eps, real_images) + tf.math.multiply((1 - eps), generated_images)

        # Combine them with real images
        combined_images = tf.concat([generated_images,
                                     real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat([self.fake_label * tf.ones((batch_size, 1)),
                            self.real_label * tf.ones((batch_size, 1))], axis=0)

        # Train the discriminator to optimality
        for step in range(5):
            with tf.GradientTape(persistent=True) as tape:
                conv_out = self.coder([combined_images, alpha * tf.ones(2 * batch_size)], training=True)
                criticism = self.criticHead(conv_out, training=True)
                wgan_loss = tf.reduce_mean(tf.math.multiply(labels, criticism))

                # get grad_x(critic(interpolated_images))
                with tf.GradientTape() as xtape:
                    xtape.watch(interp_images)
                    conv_out_interp = self.coder([interp_images, alpha * tf.ones(batch_size)], training=True)
                    interp_criticism = self.criticHead(conv_out_interp, training=True)
                critic_x_grad = xtape.gradient(interp_criticism, interp_images)
                critic_x_grad = tf.reshape(critic_x_grad, [batch_size, -1])
                penalty_loss = tf.reduce_mean(tf.square(tf.norm(critic_x_grad, axis=-1) - 1))
                #                 tf.print("critic_x_grad:", critic_x_grad)
                d_loss = wgan_loss + self.gradLAMBDA * penalty_loss + self.coder.losses + self.criticHead.losses

            coder_grads = tape.gradient(d_loss, self.coder.trainable_weights)
            criticHead_grads = tape.gradient(d_loss, self.criticHead.trainable_weights)
            del tape
            self.d_optimizer.apply_gradients(zip(coder_grads, self.coder.trainable_weights))
            self.d_optimizer.apply_gradients(zip(criticHead_grads, self.criticHead.trainable_weights))

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=latent_shape)
        random_code = random_latent_vectors[:, :, :, :self.code_features]

        # Assemble labels that say "all real images"
        # This makes the generator want to create real images (match the label) since
        # we do not include an additional minus in the loss
        misleading_labels = self.real_label * tf.ones((batch_size, 1))

        # Train the generator and encoder(note that we should *not* update the weights
        # of the critic or encoder)!
        with tf.GradientTape(persistent=True) as tape:
            fakeImages = self.generator([random_latent_vectors, alpha * tf.ones(batch_size)], training=True)
            conv_out_fake = self.coder([fakeImages, alpha * tf.ones(batch_size)], training=True)
            fakeCriticism = self.criticHead(conv_out_fake, training=True)
            # code_pred = self.coderHead(conv_out_fake)
            g_loss = tf.reduce_mean(misleading_labels * fakeCriticism)
            # info_loss = tf.reduce_mean(tf.math.squared_difference(code_pred, random_code))
            total_g_loss = g_loss + self.generator.losses  # + self.infoLAMBDA * info_loss + self.coderHead.losses

        g_grads = tape.gradient(total_g_loss, self.generator.trainable_weights)
        #         q_grads = tape.gradient(total_g_loss, self.coderHead.trainable_weights)
        del tape
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        #         self.q_optimizer.apply_gradients(zip(q_grads, self.coderHead.trainable_weights))

        return {"critic_loss": -wgan_loss, "generator_loss": g_loss,
                #                 "info_loss": info_loss,
                "gradient_penalty_loss": penalty_loss}


@tf.function
def load_image(file_path, img_size=[306, 306]):
    raw = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = (img - 0.5) * 2 # normalizing the images to [-1, 1]
    return img