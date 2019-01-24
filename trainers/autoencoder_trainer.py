import os
import warnings

from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_recall_fscore_support
from keras.layers import Dense, Multiply
from keras import Input, Model
from keras.optimizers import Adam
from bases.trainer_base import TrainerBase
from keras.utils import plot_model
import numpy as np


class AutoEncoderTrainer(TrainerBase):
    def __init__(self, model, data, config):
        super(AutoEncoderTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.model = model
        self.layers_sizes = list(map(int, self.config.layer_sizes.split(",")))
        self.layers_activations = self.config.layer_activations.split(",")

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.cp_dir,
                                      '%s.weights.{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp_name),
                monitor="val_loss",
                verbose=1,
                save_best_only=True
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.tb_dir,
                write_images=True,
                write_graph=True
            )
        )
        self.callbacks.append(FPRMetricDetail())

    def pre_train(self, order):
        new_input = Input(shape=(self.layers_sizes[order], ))
        pre_training_layer = Dense(self.layers_sizes[order + 1], activation=self.layers_activations[order])(new_input)
        new_output = Dense(self.layers_sizes[order], activation=self.layers_activations[order-1])(pre_training_layer)
        new_tensor = Input(shape=(self.layers_sizes[order], ))
        dense3 = Multiply()([new_output, new_tensor])

        new_model_feature = Model(inputs=new_input, outputs=pre_training_layer)
        new_model_feature.compile(optimizer=Adam(lr=self.config.pretrain_feature_lr),
                                  loss=self.config.pretrain_feature_loss)
        new_model = Model(inputs=[new_input, new_tensor], outputs=dense3)
        new_model.compile(optimizer=Adam(lr=self.config.pretrain_net_lr),
                          loss=self.config.pretrain_net_loss)

        print(new_model.summary())
        plot_model(new_model, to_file=self.config.img_dir + "sub_model" + str(order) + ".png", show_shapes=True)
        print(new_model_feature.summary())
        plot_model(new_model_feature, to_file=self.config.img_dir + "sub_model_feature" + str(order) + ".png", show_shapes=True)


class FPRMetricDetail(Callback):
    """
    输出F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        prd_y = np.asarray(self.model.predict(val_x))

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, f_score, support = precision_recall_fscore_support(val_y, prd_y)

        for p, r, f, s in zip(precision, recall, f_score, support):
            print(" — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f - ins %s" % (f, p, r, s))


if __name__ == "__main__":
    from utils.config_utils import process_config, get_test_args
    from models.auto_encoder_nn_model import AutoEncoderNN
    from data_loaders.data_load_cotton import CottonDL

    config = process_config('/home/zhenye/Documents/modified_code/configs/cotton_ann.json')
    # 建立模型
    model = AutoEncoderNN(config)

    # 获取数据
    data_loader = CottonDL(config)
    train_data_x, train_data_y = data_loader.get_train_data_load()
    test_data_x, test_data_y = data_loader.get_test_data_load()
    validation_data_x, validation_data_y = data_loader.get_validation_data_load()

    data = [[train_data_x, train_data_y],
            [test_data_x, test_data_y],
            [validation_data_x, validation_data_y]]

    # 进行训练
    trainer = AutoEncoderTrainer(model=model, data=data, config=config)
    trainer.pre_train(1)
