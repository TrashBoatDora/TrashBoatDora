# ecoding: utf-8
import torch
import training_history.code.resnet as resnet
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math

modelpath = "./training_history/model_save/model.th"
checkpointpath = "./training_history/model_save/checkpoint.th"
model = resnet.__dict__['resnet110']()

checkpoint = torch.load(modelpath,map_location=torch.device('cpu')) 
model.load_state_dict(checkpoint['state_dict'])
model.eval()
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])

# picture category 
category = {
    0:'飛機', 1:'車', 2:'貓', 3:'飲料',4:'情侶',
    5:'狗',6:'月亮',7:'船',8:'西裝',9:'樹'
    }

# classify the picture 
def classification(imgpath,model):
    img_lb = datasets.ImageFolder(root = imgpath, transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        normalize,
    ]))
    img_loader = torch.utils.data.DataLoader(img_lb)

    for i, (input, label) in enumerate(img_loader):
        input_var = input.float()
        category_list = model(input_var)
        path = img_lb.imgs[i][0]
        #print(category_list)
        potential_class = torch.argmax(category_list)
        answer = potential_class.item()
        print(f"{path} {potential_class.item()}, {category[potential_class.item()]}")
    return {answer,path}


if __name__ == '__main__':
    [x,y] = classification(imgpath = "./pic", model = model)
    if y == 0:
        print("飛機")
    if y == 1:
        print("車")
    if y == 2:
        print("貓")
    if y == 3:
        print("咖啡")
    if y == 4:
        print("情侶")
    if y == 5:
        print("狗")
    if y == 6:
        print("月亮")
    if y == 7:
        print("船")
    if y == 8:
        print("西裝")
    if y==9:
        print("樹")