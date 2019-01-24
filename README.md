# AutoEncoderPretrainNN
A project for Auto Encoder Neural Network 
参考 https://github.com/Karllzy/DL-Project-Template.git

## 预计程序文件夹结构
下面标上#的表示已经完成，标上*"的表示正在开发

```text
├── bases
│   ├── data_loader_base.py             - 数据加载基类#
│   ├── infer_base.py                   - 预测样本（推断）基类#
│   ├── model_base.py                   - 网络结构（模型）基类#
│   ├── trainer_base.py                 - 训练模型基类#
├── configs                             - 配置文件夹*
│   └── cotton_ann.json                 - 棉花识别配置文件*
├── data_loaders                        - 数据加载文件夹#
│   ├── __init__.py
│   ├── data_load_cotton.py             - 加载棉花数据生成器#
├── experiments                         - 实验数据文件夹#
│   └── Cotton_VS_PlasticFilm           - 实验名称#
│       ├── checkpoints                 - 存储的模型和参数#
│       │   └── ...
│       ├── images                      - 图片#
│       │   └── ...
│       └── logs                        - 日志，如TensorBoard#
│           └── ...
├── infers                              - 推断文件夹*
│   ├── __init__.py
│   ├── 待定.py                          
├── main_test.py                        - 预测样本入口*
├── main_train.py                       - 训练模型入口*
├── models                              - 网络结构文件夹#
│   ├── __init__.py
│   ├── auto_encoder_nn_model.py        - 搭建网络模型#
├── requirements.txt                    - 依赖库*
├── trainers                            - 训练模型文件夹*
│   ├── __init__.py
│   ├── autoencoder_trainer.py          - 网络训练(包含pretrain)*
└── utils                               - 工具文件夹
    ├── __init__.py
    ├── config_utils.py                 - 配置工具类#
    ├── np_utils.py                     - NumPy工具类*
    ├── utils.py                        - 其他工具类#

# 总网络结构
![image](https://github.com/Karllzy/AutoEncoderPretrainNN/blob/master/trainers/experiments/cotton_ann/images/AutoEncoderNN_Model.png)


# 预训练加载的网络结构
![image]
(https://github.com/Karllzy/AutoEncoderPretrainNN/blob/master/trainers/experiments/cotton_ann/images/sub_model1.png)
