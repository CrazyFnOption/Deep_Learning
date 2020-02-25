from config import opt
import os
import torch as t
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
from config import DefaultConfig


@t.no_grad() #pytorch>=0.5
def test(**kwargs):
    opt._parse(kwargs)

    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    #data
    train_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    result = []
    for li,(data,path) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)
        score = model(input)
        probability = t.nn.functional.softmax(score, dim=1)[:,0].detach().tolist()

        batch_results = [(path_.item(), probability_) for path_,probability_ in zip(path, probability)]
        result += batch_results
    write_csv(result, opt.result_file)
    return result

def write_csv(result, file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(result)

def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env, port=opt.vis_port)
    # 构建模型
    model = getattr(models,opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # 处理数据
    train_data = DogCat(opt.train_data_root,train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_data_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # 优化器与损失交叉变化
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr # 由于参数变化的时候 学习率也会发生变化 所以这里最好将拷贝放到里面去
    optim = model.get_optimizer(lr, opt.weight_decay)

    # 测量模型
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    # 开始训练
    for epoch in range(opt.max_epoch):
        # 下面这两个其实就是用来描述每一步需要相加的loss值 以及每次需要相加的平方
        loss_meter.reset()
        confusion_matrix.reset()
        print("hello" + str(epoch))
        for ii,(data,label) in tqdm(enumerate(train_data_loader)):
            input = data.to(opt.device)
            target = label.to(opt.device)

            optim.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optim.step()

            loss_meter.add(loss.item())
            confusion_matrix.add(score.detach(), target.detach())
            print("here ...." + str(ii))
            if (ii + 1) % opt.print_freg == 0:
                vis.plot('loss', loss_meter.value()[0])
                print("there" + str(ii))

                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        model.save()
        # validate and visualize
        val_cm,val_accuracy = val(model,val_data_loader)

        vis.plot('val_accuracy',val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch = epoch,loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr))

        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optim.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]

@t.no_grad()
def val(model,dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy

def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    train()