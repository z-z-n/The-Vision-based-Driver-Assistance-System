import os
import random

pngfilepath = r'../datasets/kitti/images/train'
saveBasePath = r"../datasets/kitti/labels"

temp_png = os.listdir(pngfilepath)
total_png = []
for png in temp_png:
    if png.endswith(".png"):
        total_png.append(png)

train_percent = 0.7
val_percent = 0.15
test_percent = 0.15

num = len(total_png)
# train = random.sample(num,0.9*num)
list = list(range(num))

num_train = int(num * train_percent)
num_val = int(num * val_percent)


train = random.sample(list, num_train)
num1 = len(train)
for i in range(num1):
    list.remove(train[i])

val_test = [i for i in list if not i in train]
val = random.sample(val_test, num_val)
num2 = len(val)
for i in range(num2):
    list.remove(val[i])

ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')

for i in train:
    name = './images/' + total_png[i][:-4] + '.png' + '\n'
    ftrain.write(name)

for i in val:
    name = './images/' + total_png[i][:-4] + '.png' + '\n'
    fval.write(name)

for i in list:
    name = './images/' + total_png[i][:-4] + '.png' + '\n'
    ftest.write(name)

ftrain.close()
fval.close()
ftest.close()