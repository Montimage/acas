from datetime import datetime

import tensorflow as tf
from keras.layers import concatenate
from keras.utils.vis_utils import plot_model
import sys
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.regularizers import L1, L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.python.keras.layers import Dropout, BatchNormalization, LeakyReLU
from tensorflow.python.keras.regularizers import L2
from tensorflow.keras.models import save_model
from tensorflow import stack

sys.path.append(sys.path[0] + '/..')
from tools.plotting import drawLossAccuracy

feature_num = 61
# exec

nb_epoch_sae = 10 # 30#10000
batch_size_sae = 16#128
nb_epoch_cnn = 5
batch_size_cnn = 32

datenow = datetime.now()

# autoencoder_norm = makedAE_simple(input_dim, 'hl_norm', 'ae_norm')
# autoencoder_mal = makedAE_simple(input_dim, 'hl_mal', 'ae_mal')

def trainSAE_CNN(x_train_norm, x_train_mal, x_train, y_train, nb_epoch_cnn, nb_epoch_sae, batch_size_cnn,batch_size_sae,datenow, save=True):
    input_dim = x_train.shape[1]
    act_reg = L1L2()
    act = LeakyReLU()
    hidden_dim_1 = input_dim
    hidden_dim_2 = input_dim
    optimizer = Adam()  # learning_rate=0.001)

    filter1 = 32
    filter2 = 64
    filter3 = 128

    input_layer_aen = Input(shape=(input_dim,))
    encoder_norm = Dense(hidden_dim_1, activation=act, use_bias=False, activity_regularizer=act_reg, name="hl_norm1")(input_layer_aen)
    encoder_norm = BatchNormalization()(encoder_norm)
    encoder_norm = Dropout(0.5)(encoder_norm)

    encoder_norm = Dense(hidden_dim_2, activation=act, use_bias=False, activity_regularizer=act_reg,name="hl_norm2")(encoder_norm)
    encoder_norm = BatchNormalization()(encoder_norm)
    encoder_norm = Dropout(0.5)(encoder_norm)

    decoder_norm = Dense(hidden_dim_1, activation=act, activity_regularizer=act_reg)(encoder_norm)  # or softmax
    decoder_norm = BatchNormalization()(decoder_norm)
    decoder_norm = Dropout(0.5)(decoder_norm)

    decoder_norm = Dense(input_dim, activation='sigmoid')(decoder_norm)  # or softmax

    autoencoder_norm = Model(inputs=input_layer_aen, outputs=decoder_norm, name="ae_norm")


    ######

    input_layer_aem = Input(shape=(input_dim,))
    encoder_mal = Dense(hidden_dim_1, activation=act, use_bias=False, activity_regularizer=act_reg, name="hl_mal1")(input_layer_aem)
    encoder_mal = BatchNormalization()(encoder_mal)
    encoder_mal = Dropout(0.5)(encoder_mal)

    encoder_mal = Dense(hidden_dim_2, activation=act, use_bias=False, activity_regularizer=act_reg, name="hl_mal2")(encoder_mal)
    encoder_mal = BatchNormalization()(encoder_mal)
    encoder_mal = Dropout(0.5)(encoder_mal)

    decoder_mal = Dense(hidden_dim_1, activation=act, activity_regularizer=act_reg)(encoder_mal)  # or softmax
    decoder_mal = BatchNormalization()(decoder_mal)
    decoder_mal = Dropout(0.5)(decoder_mal)

    decoder_mal = Dense(input_dim, activation='sigmoid')(decoder_mal)  # or softmax

    autoencoder_mal = Model(inputs=input_layer_aem, outputs=decoder_mal, name="ae_mal")

    ##############

    autoencoder_norm.compile(metrics=['accuracy', Precision(), Recall()], loss='mse', optimizer=optimizer)
    autoencoder_norm.summary()

    autoencoder_mal.compile(metrics=['accuracy', Precision(), Recall()], loss='mse', optimizer=optimizer)
    autoencoder_mal.summary()

    history_norm = autoencoder_norm.fit(x_train_norm, x_train_norm, epochs=nb_epoch_sae, batch_size=batch_size_sae,
                                        # validation_data=(x_test_norm, x_test_norm),
                                        # callbacks=[EarlyStopping(monitor="val_loss", patience=25, mode="min")]
                                        callbacks=[EarlyStopping(monitor="accuracy", patience=25, mode="max")]
                                        )
    # drawLossAccuracy(history_norm,
    #                  accuracy_file="results/sae-cnn/sae-cnn_norm_acc_{}.png".format(
    #                      datenow.strftime("%Y-%m-%d_%H-%M-%S")),
    #                  loss_file="results/sae-cnn/sae-cnn_norm_loss_{}.png".format(datenow.strftime("%Y-%m-%d_%H-%M-%S")))

    history_mal = autoencoder_mal.fit(x_train_mal, x_train_mal, epochs=nb_epoch_sae, batch_size=batch_size_sae,
                                      # validation_data=(x_test_mal, x_test_mal),
                                      # callbacks=[EarlyStopping(monitor="val_loss", patience=25, mode="min")]
                                      callbacks=[EarlyStopping(monitor="accuracy", patience=25, mode="max")]
                                      )
    # drawLossAccuracy(history_mal,
    #                  accuracy_file="results/sae-cnn/sae-cnn_mal_acc_{}.png".format(
    #                      datenow.strftime("%Y-%m-%d_%H-%M-%S")),
    #                  loss_file="results/sae-cnn/sae-cnn_mal_loss_{}.png".format(datenow.strftime("%Y-%m-%d_%H-%M-%S")))

    inp = Input(shape=(input_dim,))

    ae_norm_output = autoencoder_norm(inp)
    ae_mal_output = autoencoder_mal(inp)

    # concat_layer = concatenate([ae_norm_output, ae_mal_output, autoencoder_norm.get_layer('hl_norm2').output,autoencoder_mal.get_layer('hl_mal2').output], axis=-1)
    concat_layer = concatenate([ae_norm_output, ae_mal_output], axis=-1)
    concat_layer = tf.reshape(concat_layer, [-1, concat_layer.shape[1], 1])
    # model_j = tf.keras.models.Model(inputs=[inp], outputs=[autoencoder_norm(inp), autoencoder_mal(inp)])
    # concat_layer = Concatenate(axis=0)(model_j.output)
    # ccc = tf.reshape(concat_layer, [-1, concat_layer.shape[1], 1])

    # concat_layer = Concatenate(axis=0)([autoencoder_norm.output,
    #                                     autoencoder_norm.get_layer('hl_norm2').output,
    #                                     autoencoder_mal.output,
    #                                     autoencoder_mal.get_layer('hl_mal2').output])

    # ccc = tf.reshape(concat_layer, [-1, concat_layer.shape[1], 1])


    y = Conv1D(filter1, 1, activation=act, activity_regularizer=act_reg, kernel_regularizer=L2(),
               kernel_initializer='he_normal', input_shape=[input_dim, 1])(concat_layer)  # kernel_initializer=’he_normal’
    y = Conv1D(filter1, 1, activation=act, activity_regularizer=act_reg, kernel_regularizer=L2(),
               kernel_initializer='he_normal')(y)
    y = MaxPooling1D(2, 2)(y)
    y = Dropout(0.5)(y)

    y = Conv1D(filter2, 1, activation=act, activity_regularizer=act_reg, kernel_regularizer=L2(),
               kernel_initializer='he_normal')(y)
    y = Conv1D(filter2, 1, activation=act, activity_regularizer=act_reg, kernel_regularizer=L2(),
               kernel_initializer='he_normal')(y)
    y = MaxPooling1D(2, 2)(y)
    y = Dropout(0.5)(y)

    y = Conv1D(filter3, 1, activation=act, activity_regularizer=act_reg, kernel_regularizer=L2(),
               kernel_initializer='he_normal')(y)
    y = Conv1D(filter3, 1, activation=act, activity_regularizer=act_reg, kernel_regularizer=L2(),
               kernel_initializer='he_normal')(y)
    y = MaxPooling1D(2, 2)(y)
    y = Dropout(0.5)(y)

    y = Flatten()(y)
    y = Dense(512)(y)
    outputs = Dense(1, activation='sigmoid')(y)  # 2 = number_of_classes for sofrmax + cross entropy; 1 => sigmoid + binary cross entropy

    # cnn = Model(inputs=[model_j.output], outputs=outputs)

    # cnn = Model(inputs=concat_layer, outputs=outputs)
    cnn = Model(inputs=inp, outputs=outputs)

    cnn.compile(metrics=['accuracy', Precision(), Recall()], optimizer=optimizer, loss='binary_crossentropy')
    cnn.summary()


    print("CNN training")
    history_cnn = cnn.fit(x=x_train, y=y_train, epochs=nb_epoch_cnn, shuffle=True, batch_size=batch_size_cnn,
                          # validation_data=(x_test, y_test),
                          # callbacks=[EarlyStopping(monitor="val_loss", patience=25, mode="min")])
                          callbacks=[EarlyStopping(monitor="accuracy", patience=int(nb_epoch_cnn / 3), mode="max")])
    # drawLossAccuracy(history_cnn,
    #                  accuracy_file="results/sae-cnn/sae-cnn_acc_{}.png".format(datenow.strftime("%Y-%m-%d_%H-%M-%S")),
    #                  loss_file="results/sae-cnn/sae-cnn_loss_{}.png".format(datenow.strftime("%Y-%m-%d_%H-%M-%S")))

    print("Saving")

    if(save):
        autoencoder_norm.save('saved_models/sae_cnn_norm_{}.h5'.format(datenow.strftime("%Y-%m-%d_%H-%M-%S")))
        autoencoder_mal.save('saved_models/sae_cnn_mal_{}.h5'.format(datenow.strftime("%Y-%m-%d_%H-%M-%S")))
        cnn.save('saved_models/sae_cnn_{}.h5'.format(datenow.strftime("%Y-%m-%d_%H-%M-%S")))


    # inp = Input(shape=(input_dim,))

    # model_j = tf.keras.models.Model(inputs=[inp], outputs=[autoencoder_norm(inp), autoencoder_mal(inp)])


    # model_total = Model(inputs=[input_layer], outputs=[outputs], name='model_total')
    # model_total = Model(inputs=[model_j.input], outputs=[outputs], name='model_total')

    # model_total.save('saved_models/sae_total_{}.h5'.format(datenow.strftime("%Y-%m-%d_%H-%M-%S")))
    # plot_model(cnn, to_file='./sae_cnn_total.png', show_shapes=True, show_layer_names=True, expand_nested=True)
    return cnn

    # y_pred = cnn.predict([x_test])
    # y_pred = numpy.transpose(np.round(y_pred)).reshape(y_pred.shape[0], )
    # print(classification_report(y_test, y_pred))
    # d=datetime.now()
    # saveConfMatrix(y_true=y_test, y_pred=y_pred,
    #                filepath_csv='results/sae-cnn/conf_matrix_sae_test-cnn_{}.csv'.format(d.strftime("%Y-%m-%d_%H-%M-%S")),
    #                filepath_png='results/sae-cnn/conf_matrix_sae-cnn_{}.jpg'.format(d.strftime("%Y-%m-%d_%H-%M-%S")))
    # saveScores(y_true=y_test, y_pred=y_pred, filepath='results/sae-cnn/stats_{}'.format(d.strftime("%Y-%m-%d_%H-%M-%S")))
    #
    #
    # preds = np.array([y_pred]).T
    # res = np.append(x_test,preds,axis=1)
    # pd.DataFrame(res).to_csv("results/sae-cnn/predictions_{}.csv".format(d.strftime("%Y-%m-%d_%H-%M-%S")), index=False, header=test_data.columns)
