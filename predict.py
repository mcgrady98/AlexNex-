import json

import torch
from PIL import Image
from torchvision import transforms
from main import AlexNet


def main():
    device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

    datatrans = transforms.Compose(
                                    [transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    imgpath = 'hua.jpg'
    img =Image.open(imgpath)
    img.show()

    datapr = datatrans(img)
    img1 = torch.unsqueeze(datapr, dim=0)#将输入的数据升维度 以便送入模型

    with open('calss.json','r') as f:
        calss_indict = json.load(f)



    model = AlexNet(num_class=5).to(device) #creat model
    model_wight_path = 'canshu'
    model.load_state_dict(torch.load(model_wight_path))

    with torch.no_grad():
        output = torch.squeeze(model(img1.to(device)))
        calss  = torch.softmax(output,dim=0)
        predict = torch.argmax(calss).numpy()
        print(calss,predict)
        print("预测图片目标为：{}".format(calss_indict[str(predict)]))
if __name__ == '__main__':
    main()