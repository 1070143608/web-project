import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
import os


def process_obj(path):
    label_dict = {"gem": 0, "gem_fang": 1, "gem_square": 2, "gem_xie": 3, "other": 4}
    label = []
    all_vertices = []
    vertices = []
    flag = 0
    with open(path, 'r') as f:
        for line in f:
            string = line.split()
            if string[0] == "o":
                if flag != 0:
                    all_vertices.append(vertices)
                vertices = []
                flag = 1
                if 'gem_fang' in string[1]:
                    label.append(label_dict['gem_fang'])
                elif 'gem_square' in string[1]:
                    label.append(label_dict['gem_square'])
                elif 'gem_xie' in string[1]:
                    label.append(label_dict['gem_xie'])
                elif 'gem' in string[1]:
                    label.append(label_dict['gem'])
                else:
                    label.append(label_dict['other'])
            elif string[0] == "v":
                vertices.append([float(string[1]), -float(string[3]), float(string[2])])

        all_vertices.append(vertices)
        label = np.array(label).reshape(-1, 1)
        return all_vertices, label


def generate_graph(all_vertices, save_path):
    pca = PCA(n_components=2)
    for i in range(len(all_vertices)):
        vertices_2d = pca.fit_transform(all_vertices[i])
        x = []
        y = []
        for j in range(len(vertices_2d)):
            x.append(vertices_2d[j][0])
            y.append(vertices_2d[j][1])
        plt.scatter(x, y)
        plt.axis("off")
        plt.savefig(save_path + "/" + str(i) + ".jpg", dpi=10)
        plt.close()
    print("Successfully get features of all products.")


def net(shape):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, 3, padding='same', input_shape=shape, activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Conv2D(64, 5, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(5, activation='softmax'))

    return model


train_path = "gemlib.obj"
test_path = "gemtest.obj"
train_jpg_path = "./train"
test_jpg_path = "./test"
image_size = [64, 48]

width = image_size[0]
height = image_size[1]

train_all_vertices, train_y = process_obj(train_path)
test_all_vertices, test_y = process_obj(test_path)

generate_graph(train_all_vertices, train_jpg_path)
generate_graph(test_all_vertices, test_jpg_path)

train_file_list = os.listdir(train_jpg_path)
train_size = len(train_file_list)
test_file_list = os.listdir(test_jpg_path)
test_size = len(test_file_list)

train_x = np.zeros((train_size, height, width, 3))
test_x = np.zeros((test_size, height, width, 3))

for i in range(train_size):
    img = Image.open(train_jpg_path + "/" + str(i) + ".jpg")
    arr = np.asarray(img)
    train_x[i, :, :, :] = arr / 255
for i in range(test_size):
    img = Image.open(test_jpg_path + "/" + str(i) + ".jpg")
    arr = np.asarray(img)
    test_x[i, :, :, :] = arr / 255

train_y_onehot = keras.utils.to_categorical(train_y, 5)
model = net(train_x.shape[1:])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y_onehot, batch_size=10, epochs=30)
model.save('./saved_weights')
model.evaluate(test_x, keras.utils.to_categorical(test_y, 5), batch_size=10)
# load上次训练好的权重，并再次训练
# model = keras.models.load_model('./saved_weights')
# model.fit(train_x, train_y_onehot, batch_size=10, epochs=30)
# model.save('./saved_weights')
prediction_onehot = model.predict(test_x)
prediction = np.argmax(prediction_onehot, axis=1)

print(np.reshape(test_y, (1, -1))[0])
print(prediction)

