class DefaultConfig(object):
    env = 'default'   # visdom 环境
    model = 'CaffeeNet' # 使用的模型，名字必须与models/__init__.py中的名字一致
    
    train_data_root = './data/images' # 训练集存放路径
    test_data_root = './data/images' # 测试集存放路径
    load_model_path = None # 加载预训练的模型的路径，为None代表不加载

    batch_size = 3  # 显存4G（batch_size=56约需8G显存） 
    use_gpu = False   # use GPU or not
    num_workers = 4  # 4线程加载数据
    print_freq = 10  # 每 N batch 输出信息

    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
      
    max_epoch = 10
    lr = 0.001         # initial learning rate 最好不要太大
    lr_decay = 0.95     # 衰减因子, lr = lr*lr_decay
    weight_decay = 1e-4 # 损失函数
    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        # 更新配置参数
        for k, v in kwargs.iteritems():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self, k, v)
            
        # 打印配置信息	
        print('user config:')
        for k, v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                print(k, getattr(self, k))
