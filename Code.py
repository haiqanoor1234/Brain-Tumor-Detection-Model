import os
import io
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt1_conf
import matplotlib.pyplot as plt1_roc
import itertools
from itertools import cycle
import pickle
from matplotlib.pyplot import imshow
from keras.models import model_from_json
from PIL import Image

plt.style.use('dark_background')


class CNN_Model:
    def __init__(self):
        self.test_image = None
        self.model = None
        self.output=None

    def fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def fileDialog(self, filename):
        self.test_image = Image.open(filename)
        self.test_image = self.test_image.resize((128, 128))
        self.RunTest()

    def TrainModel(self):
        encoder = OneHotEncoder()
        encoder.fit([[0], [1]])
        data = []
        paths = []
        result = []

        for r, d, f in os.walk(r"C:\Users\DELL\Desktop\FYP\Model\archive\yes"):
            for file in f:
                if '.jpg' in file:
                    paths.append(os.path.join(r, file))

        for path in paths:
            img = Image.open(path)
            img = img.resize((128, 128))
            img = np.array(img)
            if img.shape == (128, 128, 3):
                data.append(np.array(img))
                result.append(encoder.transform([[0]]).toarray())

        paths = []
        for r, d, f in os.walk(r"C:\Users\DELL\Desktop\FYP\Model\archive\no"):
            for file in f:
                if '.jpg' in file:
                    paths.append(os.path.join(r, file))

        for path in paths:
            img = Image.open(path)
            img = img.resize((128, 128))
            img = np.array(img)
            if img.shape == (128, 128, 3):
                data.append(np.array(img))
                result.append(encoder.transform([[1]]).toarray())
        data = np.array(data)
        result = np.array(result)
        result = result.reshape(139, 2)
        n_classes = result.shape[1]
        print("CLASSES")
        print(n_classes)

        x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.3, shuffle=True, random_state=0)

        # self.model = Sequential()
        #
        # self.model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding='Same'))
        # self.model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', padding='Same'))
        #
        # self.model.add(BatchNormalization())
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.25))
        #
        # self.model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='Same'))
        # self.model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='Same'))
        #
        # self.model.add(BatchNormalization())
        # self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # self.model.add(Dropout(0.25))
        #
        #
        # self.model.add(Flatten())
        #
        # self.model.add(Dense(512, activation='relu'))
        # self.model.add(Dropout(0.5))
        #
        # self.model.add(Dense(2, activation='softmax'))
        #
        # print(self.model.summary())

        # self.model.compile(loss="categorical_crossentropy", optimizer='Adamax', metrics=['accuracy'])
        # hist=self.model.fit(x_train, y_train, epochs=30, batch_size=40, verbose=1, validation_data=(x_test, y_test))
        # self.model.evaluate(x_test, y_test, verbose=0)


        # f = open('history.pckl', 'wb')
        # pickle.dump(hist.history, f)
        # f.close()


        def plot_confusion_matrix(cm, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues):
            plt1_conf.figure(figsize=(6, 6))
            plt1_conf.imshow(cm, interpolation='nearest', cmap=cmap)
            plt1_conf.title(title)
            plt1_conf.colorbar()
            tick_marks = np.arange(len(classes))
            plt1_conf.xticks(tick_marks, classes, rotation=90)
            plt1_conf.yticks(tick_marks, classes)
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            thresh = cm.max() / 2.
            cm = np.round(cm, 2)
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            plt1_conf.tight_layout()
            plt1_conf.ylabel('True label')
            plt1_conf.xlabel('Predicted label')
            fig3 = plt1_conf.gcf()
            plot_confuse = self.fig2img(fig3)
            plt1_conf.savefig("plot4.png", bbox_inches='tight')
            plt1_conf.clf()

            return plot_confuse

        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        # # load existing model into new model
        self.model=keras.models.load_model("model.h5")
        self.model.compile(loss="categorical_crossentropy", optimizer='Adamax', metrics=['accuracy'])
        self.model.evaluate(x_test , y_test , verbose=0)
        print(self.model.summary())

        f = open('history.pckl', 'rb')
        hist = pickle.load(f)
        f.close()

        y_predict = self.model.predict(x_test)
        predict_label1 = np.argmax(y_predict, axis=-1)
        true_label1 = np.argmax(y_test, axis=-1)
        CN = confusion_matrix(true_label1, predict_label1)
        tn, fp, fn, tp = CN.ravel()
        print(CN)
        img3 = plot_confusion_matrix(CN, classes=['no', 'yes'], normalize=False)

        fpr = dict()
        tpr = dict()
        roc_auc1 = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predict[:, i])
            roc_auc1[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_predict.ravel())
        roc_auc1["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc1["macro"] = auc(fpr["macro"], tpr["macro"])

        plt1_roc.figure()
        plt1_roc.plot(fpr["micro"],
                      tpr["micro"],
                      label="micro-average ROC curve (area = {0:0.2f}".format(roc_auc1["micro"]),
                      color="deeppink",
                      linestyle=":",
                      linewidth=4, )
        plt1_roc.plot(fpr["macro"],
                      tpr["macro"],
                      label="macro-average ROC curve (area = {0:0.2f}".format(roc_auc1["macro"]),
                      color="navy",
                      linestyle=":",
                      linewidth=4, )
        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(n_classes), colors):
            plt1_roc.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc1[i])
            )

        plt1_roc.plot([0, 1], [0, 1], "k--", lw=2)
        plt1_roc.xlim([0.0, 1.0])
        plt1_roc.ylim([0.0, 1.05])
        plt1_roc.xlabel("Fasle Positive Rate")
        plt1_roc.ylabel("True Positive Rate")
        plt1_roc.title("ROC Curve")
        plt1_roc.legend(loc="lower right")

        fig2 = plt1_roc.gcf()
        img2 = self.fig2img(fig2)
        plt1_roc.savefig("plot3.png", bbox_inches='tight')
        plt1_roc.clf()

        # model_json = self.model.to_json()
        # with open("model.json", "w") as json_file:
        #     json_file.write(model_json)
        # self.model.save("model.h5")

        plt.plot(hist["loss"])
        plt.plot(hist["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Test", "Validation"], loc="upper right")

        fig = plt.gcf()
        img = self.fig2img(fig)
        plt.savefig("plot1.png", bbox_inches='tight')
        plt.clf()

        plt1.plot(hist["accuracy"], c="purple")
        plt1.plot(hist["val_accuracy"], c="orange")
        plt1.title("Accuracy")
        plt1.ylabel("Accuracy")
        plt1.xlabel("Epochs")
        plt1.legend(["train", "test"])


        fig1 = plt1.gcf()
        img1 = self.fig2img(fig1)
        plt1.savefig("plot2.png", bbox_inches='tight')
        plt1.clf()


        return img, img1, img2, img3

    def names(self, number):
        if number == 0:
            return 'Its a Tumor'
        else:
            return 'No, Its not a tumor'

    def RunTest(self):
        img1 = self.test_image
        x = np.array(img1.resize((128, 128)))
        x = x.reshape(1, 128, 128, 3)
        res = self.model.predict_on_batch(x)
        classification = np.where(res == np.amax(res))[1][0]
        self.output = str(res[0][classification] * 100) + '% Confidence That ' + self.names(classification)
        print(self.output)
    
    def classifyImage(self,img):
        x = np.array(img.resize((128, 128)))
        x = x.reshape(1, 128, 128, 3)
        res = self.model.predict_on_batch(x)
        classification = np.where(res == np.amax(res))[1][0]
        output = str(res[0][classification] * 100) + '% Confidence That ' + self.names(classification)
        return output
    