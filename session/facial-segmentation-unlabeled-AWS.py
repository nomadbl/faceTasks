import subprocess
import argparse
from unet_segmentor_lib import *
from gan_segmentor_lib import *

print(tf.version.VERSION)
subprocess.run(["nvidia-smi", "-L"])

# uncomment for debugging layers
# tf.config.run_functions_eagerly(True) 


def main_unet(station):
    # station = "home"
    # station = "aws"
    train = False
    evaluate = False

    home = None
    if station == "aws":
        home = os.getenv("HOME")
    elif station == "home":
        home = "/home/lior/PycharmProjects/facesTasks"
    
    print("current working directory:", home)

    copies_per_image = 1
    images_per_batch = 3
    dataset = tf.data.Dataset.list_files(f'{home}/images/*.jpg')
    # train/test split
    image_count = tf.cast(dataset.cardinality(), tf.float32)
    train_perc = tf.constant(0.8)
    train_dataset = dataset.take(tf.cast(tf.math.round(image_count * train_perc), tf.int64))
    val_dataset = dataset.skip(tf.cast(tf.math.round(image_count * train_perc), tf.int64))
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    val_dataset = val_dataset.shuffle(buffer_size=1000)
    # train several times on each image (augmentations will be different)
    # train_dataset = train_dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x)#.repeat(copies_per_image)
    #                                          ,cycle_length=4, block_length=copies_per_image)
    # val_dataset = val_dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x)#.repeat(copies_per_image)
    #                                      ,cycle_length=4, block_length=copies_per_image)
    # read the images
    train_dataset = train_dataset.map(load_image, num_parallel_calls=3).cache()
    train_dataset = train_dataset.map(partial(tf.image.convert_image_dtype, dtype=tf.float32), num_parallel_calls=3)
    val_dataset = val_dataset.map(load_image, num_parallel_calls=3).cache()
    val_dataset = val_dataset.map(partial(tf.image.convert_image_dtype, dtype=tf.float32), num_parallel_calls=3)

    # prepare augmented images. dataset of (augmented image, image), block_length augmentions for each
    train_dataset = train_dataset.map(partial(random_crop, size=128), num_parallel_calls=3)
    train_dataset = train_dataset.map(partial(box_delete, size=32, l=0.06), num_parallel_calls=3)

    val_dataset = val_dataset.map(partial(random_crop, size=128), num_parallel_calls=3)
    val_dataset = val_dataset.map(partial(box_delete, size=32, l=0.06), num_parallel_calls=3)

    # visualize
    train_dataset.apply(visualize_training)

    # save current state of pipeline for segmentation ahead
    train_dataset_seg = train_dataset
    val_dataset_seg = val_dataset

    # flatten the original image ("label") so we can use sample weights
    train_dataset = train_dataset.map(partial(flatten_labels), num_parallel_calls=3)
    val_dataset = val_dataset.map(partial(flatten_labels), num_parallel_calls=3)
    # shuffle and batch
    train_dataset = train_dataset.batch(images_per_batch * copies_per_image, drop_remainder=True)
    val_dataset   = val_dataset.batch(images_per_batch * copies_per_image, drop_remainder=True)
    # prefetch
    train_dataset = train_dataset.prefetch(10)
    val_dataset = val_dataset.prefetch(4)

    print("done dataset preparations")
    print(train_dataset)

    # AE V1
    # bridge_features = [16]
    # encoder_filters_list = [ 64, 64, 128, 128, 256, 256]
    # decoder_filters_list = [256, 256, 128, 128, 64, 64]
    # head_filters_list = [32, 32, 16, 3]
    bridge_features = [16, 16, 16, 16]
    encoder_filters_list = [128, 64, 32, 32, 32]
    skip_connections =     [1 ,  1 , 1,  0,  0]
    decoder_filters_list = [32, 32, 32,  64, 128]
    head_filters_list = [3]

    cp_file = f'{home}/session/model_cp.h5'
    if os.path.exists(cp_file):
        print("found checkpoint, loading")
        model = tf.keras.models.load_model(cp_file,
                                   custom_objects={'ConvBlock': ConvBlock, 'DeConvBlock': DeConvBlock,
                                                   'PadToSize': PadToSize, 'PixelWiseHuber': PixelWiseHuber,
                                                   'PaddedConcat': PaddedConcat, 'PaddedAdd': PaddedAdd})
        # test model
        if evaluate:
            print("evaluating model on val_dataset:")
            results = model.evaluate(val_dataset)
            print(dict(zip(model.metrics_names, results)))

    else:
        print("checkpoint not found, compiling fresh model")
        model = get_completion_model(bridge_features, encoder_filters_list, decoder_filters_list, skip_connections,
                                     head_filters_list)
    #     model = get_simple_model() # for debugging memory leak and comparison with our model
        opt = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=0.001) # default lr, use reduction strategy below
        model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

    # garbage_collect_cb = GCCallback()
    # lr_strategy_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.1, patience=3, verbose=1,
                                                            # mode="min", min_delta=0.001, cooldown=0, min_lr=0.0000001)

    model.summary()

    if not station == "home" and train is True:  # don't train on home pc
        history = model.fit(x=train_dataset, # steps_per_epoch=2000,
                            epochs=100, validation_data=val_dataset, validation_steps=5, validation_freq=1,
                            workers=3
                            , callbacks=[tf.keras.callbacks.ModelCheckpoint(cp_file, verbose=1)]
                            )

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig(f'{home}/train_history.png')

    # visualize results

    vis_val = val_dataset.unbatch().shuffle(buffer_size=100).shard(num_shards=100, index=0)

    do_my_model = get_apply_model(model)
    results_val = vis_val.map(do_my_model)
    print(results_val)
    results_val.apply(visualize_results)

    # We can use this model as a segmentation model by removing the head and using a clustering algorithm.

    # We cluster the pixel features using yet another fully connected auto encoder with shared weights (1x1 convs).
    # The loss function is L=|p-D(E(p))|^2. The code is analogous to a ICA (independent component analysis).
    # Minimizing the loss function is equivalent to maximizing the mutual information between the code and the original
    # pixel, which is just the ICA analysis.
    #
    # f:features
    #
    # c:code
    #
    # I(f;c)=H(f) - H(f|c)
    #
    # f = D(c) + n
    #
    # P(f|c) = P(n) = P_D(p-D(c)) = P(L)
    #
    # L = |p-D(c)|^2
    #
    # argmax_D I = argmin_D L
    #
    # The autoencoder can be linear (ICA) or nonlinear (deep).
    # The code can be visualised using colormaps

    layer_name = "pad_to_size_1"
    seg_model = keras.Model(inputs=model.input,
                            outputs=model.get_layer(layer_name).output)
    # seg_model.trainable = False # freezing not needed since the models are not actually connected

    features_img = tf.keras.Input(shape=[128, 128, 128], dtype='float32', name="input_org")
    code = keras.layers.Conv2D(filters=256, kernel_size=(1,1), data_format='channels_last')(features_img)
    code = keras.layers.ReLU()(code)
    code = keras.layers.Conv2D(filters=64, kernel_size=(1,1), data_format='channels_last')(code)
    code = keras.layers.ReLU()(code)
    code = keras.layers.Conv2D(filters=16, kernel_size=(1,1), data_format='channels_last')(code)
    code = keras.layers.ReLU(name="encoder_out")(code)
    decoder = keras.layers.Conv2D(filters=64, kernel_size=(1,1), data_format='channels_last')(code)
    decoder = keras.layers.ReLU()(decoder)
    decoder = keras.layers.Conv2D(filters=256, kernel_size=(1,1), data_format='channels_last')(decoder)
    decoder = keras.layers.ReLU()(decoder)
    decoder = keras.layers.Conv2D(filters=128, kernel_size=(1,1),
                                  data_format='channels_last', name="decoder_out")(decoder)

    autoencoder_model = keras.Model(inputs=features_img, outputs=decoder)

    def get_gt(image, gt, w):
        return gt

    print("building clustering training pipline")

    def extract_seg_features(images):
        return do_seg_model(images, seg_model)

    # training data pipeline
    # shuffle and batch
    images_per_batch = 10
    train_dataset_seg_batched = train_dataset_seg.batch(images_per_batch, drop_remainder=True)
    val_dataset_seg_batched   = val_dataset_seg.batch(images_per_batch, drop_remainder=True)

    train_dataset_seg_pixels = train_dataset_seg_batched.map(get_gt, num_parallel_calls=3)
    val_dataset_seg_batched  = val_dataset_seg_batched.map(get_gt, num_parallel_calls=3)
    train_dataset_seg_pixels = train_dataset_seg_pixels.map(extract_seg_features, num_parallel_calls=3)
    val_dataset_seg_batched  = val_dataset_seg_batched.map(extract_seg_features, num_parallel_calls=3)

    print("pca_train_dataset:",train_dataset_seg_pixels)
    # process 1000 pixels per batch instead of ~128 (memory effcient)
    autoencoder_train_dataset = train_dataset_seg_pixels.map(lambda x: (x,x)).prefetch(10)
    print("train_dataset_seg_pixels:",train_dataset_seg_pixels)

    ae_cp_file = f'{home}/session/autoencoder_deep_cp.h5'
    training = False
    if os.path.exists(ae_cp_file):
        print("found checkpoint, loading")
        load_ae = True
    else:
        load_ae = False
    if load_ae and not training:
        autoencoder_model = tf.keras.models.load_model(ae_cp_file)

    if training:
        opt = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=0.001) # default lr, use reduction strategy below
        autoencoder_model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
        if load_ae:
            print("loading model weights for training")
            autoencoder_model.load_weights(filepath=ae_cp_file)
        autoencoder_model.fit(autoencoder_train_dataset,
                              epochs=3, validation_data=val_dataset, validation_steps=5, validation_freq=1,
                              workers=3
                              , callbacks=[tf.keras.callbacks.ModelCheckpoint(ae_cp_file, verbose=1, save_freq=500)])

    layer_name = "encoder_out"
    encoder_model =  keras.Model(inputs=autoencoder_model.input,
                                 outputs=autoencoder_model.get_layer(layer_name).output)

    val_dataset_seg_code = val_dataset_seg_batched.map(encoder_model).unbatch()
    val_dataset_seg_code = val_dataset_seg_code.map(image_float_to_int)

    val_dataset_seg_code.apply(visualize_seg)

    # We see that for the most part the code corresponds to features which are spread across the whole picture.
    # We don't see features that correspond to particular body parts like eyes etc.
    # This makes sense since the results were quite blurry and seem to be somewhat of a naive extrapolation of the
    # surroundings.
    #
    # This result is not due to the encoder since we can decode the pixels from it very precisely.
    # In fact it may be that the features (and the code) correspond to the rgb values
    # due to the UNET's skip connections.


def main_gan(station):
    # # Attempt #2
    #
    # We will use a GAN architecture similar to infoGAN: https://arxiv.org/pdf/1606.03657.pdf
    #
    # However there are several changes relative to this work:
    #
    # 1. As we want a segmentation map, we need to generate a code per pixel.
    # Thus the architecture will be similar to what we tried above with two changes.
    # We will not use skip connections between the encoder and decoder so we can clearly identify the code.
    # Also we will use "same" padding and strided convolutions to mimic the downsampling while generating a code
    # for each pixel.
    #
    # 2. As was shown just above, maximizing the mutual information is the same as minimizing the distance between
    # the decoded signal and the original signal. In infoGAN the mutual information I(c;G(c,z)) is maximized.
    # Therefore we will minimize the distance L=|c-E(G(c,z))|^2. That is, we use the encoder part on the generated
    # image to generate a code as similar as possible to the original one that was sampled while training the GAN.
    # This part can be trained seperately from the generator (i.e. decoder) and the discriminator.
    # Therefore the segmentor is actually the encoder which we will get "for free" with this method.
    #
    # 3. I will use the WGAN method of https://arxiv.org/pdf/1701.07875.pdf.
    # Additionaly we use the gradient penalty of https://arxiv.org/pdf/1704.00028.pdf
    #
    # To reduce training time the discriminator can actually be trained as an alternate "head" of the encoder.
    # This is actually what was done in the infoGAN paper.
    #
    # In drawing samples from the latent space, we use a gaussian of unit variance.
    # Since each pixel has N components in the code, we need each component to be a gaussian of variance 1/sqrt(N)
    # (this way the total variance is 1).
    # station = "home"
    # station = "aws"
    home = None
    if station == "aws":
        home = os.getenv("HOME")
    elif station == "home":
        home = "/home/lior/PycharmProjects/facesTasks"

    print("current working directory:", home)

    dataset = tf.data.Dataset.list_files(f'{home}/images/*.jpg')
    # train/test split
    image_count = tf.cast(dataset.cardinality(), tf.float32)
    train_perc = tf.constant(0.8)
    train_dataset = dataset.take(tf.cast(tf.math.round(image_count * train_perc), tf.int64))
    val_dataset = dataset.skip(tf.cast(tf.math.round(image_count * train_perc), tf.int64))
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    val_dataset = val_dataset.shuffle(buffer_size=1000)

    # read the images
    train_dataset = train_dataset.map(load_image, num_parallel_calls=3).cache()
    val_dataset = val_dataset.map(load_image, num_parallel_calls=3).cache()

    # further dataset processing inside GAN class

    print("done dataset preparations")
    print(train_dataset)

    decoder_filters_list = [64, 64, 32, 32,  16, 16, 8]
    latent_features = 215
    code_features = 16
    noise_features = latent_features - code_features
    pixel_features = 8
    info_lambda = 100
    grad_lambda = 10

    gan_cp_file = f'{home}/session/infoWGAN_cp.h5'
    cp_dir = f'{home}/session/progressive_gan/'
    training = True
    if os.path.exists(gan_cp_file):
        print("found checkpoint, loading")
        load = True
    else:
        load = False
    if load and not training:
        gan_trainer = tf.keras.models.load_model(gan_cp_file)
    else:
        gan_trainer = InfoWGAN(code_features, noise_features, pixel_features, decoder_filters_list, CP_dir=cp_dir,
                               epochs_per_phase=1)

    if training:
        # d_opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0, beta_2=0.9, epsilon=1e-05, name='Adam')
        # g_opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0, beta_2=0.9, epsilon=1e-05, name='Adam')
        # q_opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9, epsilon=1e-05, name='Adam')
        d_opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001,name='RMSprop')
        g_opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001,name='RMSprop')
        q_opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001,name='RMSprop')
        gan_trainer.compile(d_opt, g_opt, q_opt, grad_lambda, info_lambda)
    #     if load:
    #         print("loading model weights for training")
    #         gan_trainer.load_weights(filepath=gan_cp_file)

        callbacks = [
                     # tf.keras.callbacks.ModelCheckpoint(gan_cp_file, verbose=1, save_freq=500),
                     ShowGeneratorCallback(display=500),
                     # MemoryCallback()
                    ]
        gan_trainer.fit(train_dataset, epochs=3, validation_data=val_dataset, validation_steps=5, validation_freq=1,
                        workers=3, callbacks=callbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", dest="v", type=int, choices={1, 2}, default=2, required=False, help="1 - unet, 2 - gan")
    parser.add_argument("--station", dest="station", type=str, choices={"aws", "home"}, default="aws", required=False)
    args = parser.parse_args()

    if args.v == 1:
        main_unet(args.station)

    elif args.v == 2:
        main_gan(args.station)
