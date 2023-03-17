import json
import time

import torch
import torch.utils.data.dataloader
import os
from main import AlexNet
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
    print("use decice:{}".format(device))

    data_tansform = {
                    "train" :transforms.Compose([transforms.RandomResizedCrop(224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                 ]),
                    "val"  :transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
                    }



    imgpath = os.path.abspath(os.getcwd())
    train_datas = datasets.ImageFolder(root=os.path.join(imgpath,"train"), transform=data_tansform["train"])
    class_path = train_datas.class_to_idx
    class_path_1 = dict((v,k)for k,v in class_path.items())
    jos_class = json.dumps(class_path_1,indent=4)
    with open('calss.json','w') as json_file:
        json_file.write(jos_class)

    val_datas = datasets.ImageFolder(root=os.path.join(imgpath,"val"), transform=data_tansform["val"])
    val_datas_len = len(val_datas)
    train_loder = torch.utils.data.DataLoader(train_datas,batch_size=64,shuffle=True,num_workers=0)
    train_loder_len = len(train_loder)
    val_loder = torch.utils.data.DataLoader(val_datas,batch_size=64, shuffle=True, num_workers=0)
    val_loder_len = len(val_loder)

    net = AlexNet(num_class=5,init_weights=True) #网络实例化
    net.to(device)  #把模型送入到GPU
    loss_function = nn.CrossEntropyLoss() # 损失函数
    optimizer = optim.Adam(net.parameters(),lr=0.0002) #优化器
    epochs = 10
    running_loss = 0.0
    tm1 = time.perf_counter()
    for epoch in range(epochs):
        net.train()
        for step, data in enumerate(train_loder): #从train_loder 中取出 图片与标签
            img, label = data
            optimizer.zero_grad()
            output = net(img.to(device))
            loss = loss_function(output, label.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            rate = (step+1)/train_loder_len
            print('训练进度:{}% LOSS:{} 耗费时间：{}'.format(int(rate*100),loss,time.perf_counter()-tm1))

        net.eval()
        acc = 0.0
        best_acc = 0.0
        with torch.no_grad():
            for img in val_loder:
                img1, label1 = img
                outputs = net(img1.to(device))
                predict_y = torch.max(outputs,dim=1)[1]
                acc += torch.eq(predict_y,label1.to(device)).sum().item()
            racc = acc/val_datas_len
            if racc > best_acc:
                best_acc = racc
                torch.save(net.state_dict(),'canshu')    #保存模型参数
            print('epoch:{},trainloss:{},acc:{}'.format(epoch+1,running_loss,racc))


    print('训练完成')

if __name__ == '__main__':
    main()