from turtle import Turtle
import mmcv
import os.path as osp
from mmcv import Config
import time
from mmcls.apis import inference_model, init_model, show_result_pyplot, train_model
from mmcls.models import build_classifier
from mmcls.datasets import build_dataset
from mmcv.runner import load_checkpoint
import os

'''
 ToDo: 1.数据集种类选择
       2.网络结构选择
       3.是将mm系列包一起打包在内部, 还是外部下载
       4.checkpoints应该归属到每个网络文件夹下
       5.外部是否需要声明.pth还是我们预先帮他们下载好
       6.文件目录需要调整
       7.inference增加数据集选项
'''

class MMClassification:
    def __init__(self, 
        backbone=None,
        # config='./configs/mobilenet_v2/mobilenet.py',
        pretrain='MobileNet',
        checkpoints=None
                 ):

        # self.pretrain = pretrain
        # 默认的config和checkpoints后续改为ResNet
        self.config = './utils/mobilenet/mobilenet.py'
        self.checkpoint = './utils/mobilenet/mobilenet.pth'

        self.pretrain = os.path.join('./utils', pretrain)
        ckpt_cfg_list = list(os.listdir(self.pretrain))
        for item in ckpt_cfg_list:
            if item[-1] == 'y':
                self.config = os.path.join(self.pretrain, item)
            elif item[-1] == 'h':
                self.checkpoint = os.path.join(self.pretrain, item)
            else:
                print("Warning!!! There is an unrecognized file in the pretrain folder.")

        self.cfg = Config.fromfile(self.config)
        self.dataset_path = None
        self.lr = None
        self.backbonedict = {
            "MobileNet": './utils/mobilenet/mobilenet.py',
            "ResNet": 'xxxxxx.py',
            'LeNet': './utils/lenet/lenet5_mnist.py'
            # 下略
        }
        self.dataset_type_dict = {
            "ImageNet": 'ImageNet',
            "coco": 'CocoDataset',
            "voc": 'VOCDataset',
            "cifar": 'CIFAR10'
            # 下略
        }

    def train(self, random_seed=0, save_fold='./checkpoints', backbone="Resnet", distributed=False, validate=True, device="cpu",
              metric='accuracy', optimizer="SGD", epochs=100, lr=0.001, weight_decay=0.001):# 加config

        # 获取config信息

        self.cfg = Config.fromfile(self.backbonedict[backbone])
        self.load_dataset(self.dataset_path)
        print("进行了cfg的切换")
            # 进行
        self.cfg.work_dir = save_fold
        # 创建工作目录
        mmcv.mkdir_or_exist(osp.abspath(self.cfg.work_dir))
        # 创建分类器
        model = build_classifier(self.cfg.model)
        model.init_weights()

        datasets = [build_dataset(self.cfg.data.train)]

        # 添加类别属性以方便可视化
        model.CLASSES = datasets[0].CLASSES

        n_class = len(model.CLASSES) 
        if n_class <= 5:
            self.cfg.evaluation.metric_options = {'topk': (1,)}
        else:
            self.cfg.evaluation.metric_options = {'topk': (5,)}

        # 根据输入参数更新config文件
        self.cfg.optimizer.lr = lr  # 学习率
        self.cfg.optimizer.type = optimizer  # 优化器
        self.cfg.optimizer.weight_decay = weight_decay  # 优化器的衰减权重
        self.cfg.evaluation.metric = metric  # 验证指标
        self.cfg.runner.max_epochs = epochs  # 最大的训练轮次

        # 设置每 5 个训练批次输出一次日志
        self.cfg.log_config.interval = 1
        self.cfg.gpu_ids = range(1)

        self.cfg.seed = random_seed

        train_model(
            model,
            datasets,
            self.cfg,
            distributed=distributed,
            validate=validate,
            timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            device=device,
            meta=dict()
        )
        
    def inference(self, device='cpu', is_trained=False,
                image=None, show=True):

        model_fold = self.cfg.work_dir
        
        img_array = mmcv.imread(image)
        checkpoint = self.checkpoint
        if is_trained:
            checkpoint = os.path.join(model_fold, 'latest.pth')
        model = init_model(self.config, checkpoint, device=device)
        result = inference_model(model, img_array) # 此处的model和外面的无关,纯局部变量
        if show == True:
            show_result_pyplot(model, image, result)
        return result

    def load_dataset(self, path, dataset_type):
        self.dataset_path = path

        self.cfg.img_norm_cfg = dict(
            mean=[124.508, 116.050, 106.438],
            std=[58.577, 57.310, 57.437],
            to_rgb=True
        )

        self.cfg = Config.fromfile(self.dataset_type_dict[dataset_type])
        # self.cfg.dataset_type = dataset_type

        if self.dataset_type_dict == 'ImageNet':
            self.cfg.data.train.data_prefix = path + '/training_set/training_set'
            self.cfg.data.train.classes = path + '/classes.txt'

            self.cfg.data.val.data_prefix = path + '/val_set/val_set'
            self.cfg.data.val.ann_file = path + '/val.txt'
            self.cfg.data.val.classes = path + '/classes.txt'

            self.cfg.data.test.data_prefix = path + '/test_set/test_set'
            self.cfg.data.test.ann_file = path + '/test.txt'
            self.cfg.data.test.classes = path + '/classes.txt'
        


    # def print_configs(self):
    #     if self.pretrain is None:
    #         model = "MobileNet"
    #     else:
    #         model = self.pretrain
    #     print("当前网络结构：" + model)
    #     print("数据集路径：", self.dataset_path)
    #     print("学习率", self.lr)

# if __name__ == "__main__":
#     img = '../img/test.jpg'
#     # mmcls_test(img)
#     model = MMClassification()
#     result = model.inference(image=img)
#     print(result)
#     show_result_pyplot(model.SOTA_model, img, result)
