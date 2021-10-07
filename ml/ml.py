import os
from librosa.core import audio
import scipy.io
import scipy.io.wavfile
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import tempfile
import torch
import librosa
from torch.nn import CrossEntropyLoss
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import argparse
import glob
import cv2


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def open_audio(filepath):
    sampleRate, audioBuffer = scipy.io.wavfile.read(filepath)
    channel0 = list(audioBuffer[:, 0])
    return channel0


def save_as_spectrogram(input_audio_file, output_image_path, sample_rate=16000, dpi=300):
    y, sr = librosa.load(input_audio_file, sr=sample_rate, mono=False)
    fast_fourier_signal = np.abs(librosa.stft(y[0], n_fft=255, hop_length=200))
    fast_fourier_signal_log = librosa.amplitude_to_db(
        fast_fourier_signal, top_db=None)

    height = fast_fourier_signal.shape[0]
    width = fast_fourier_signal.shape[1]

    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(fast_fourier_signal_log, aspect='auto', cmap='hot')
    fig.savefig(output_image_path, dpi=dpi)
    plt.close()


class ImageDataset(Dataset):

    def __init__(self, path_to_trainfolders, img_size=224):
        self.__class_names = []
        self.__classes = []
        self.__imagepaths = []
        self.__img_size = img_size

        for path in path_to_trainfolders:
            fullpath, classname = os.path.split(path)
            self.__class_names.append(classname)
            filenames = os.listdir(path)
            self.__classes += [classname]*len(filenames)
            self.__imagepaths += [os.path.join(path, s) for s in filenames]

    @staticmethod
    def transform(image, img_size=128, padding=False):

        if(padding):
            target_size = max(image.size)
            result = Image.new('RGB', (target_size, target_size), "white")
            try:
                result.paste(image, (int(
                    (target_size - image.size[0]) / 2), int((target_size - image.size[1]) / 2)))
            except:
                print("Error on image " + image)
                raise Exception('pad_image error')
            return result

        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        image = transform(image)

        return image

    def __len__(self):
        return len(self.__imagepaths)

    def __getitem__(self, index):
        img_name = self.__imagepaths[index]
        image = Image.open(img_name)
        image = self.transform(image, self.__img_size)
        label = self.__class_names.index(self.__classes[index])
        return image, label


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def predict(audio_file, model_onnx_path):
    f = tempfile.NamedTemporaryFile()
    save_as_spectrogram(audio_file, f.name)
    net = cv2.dnn.readNetFromONNX(model_onnx_path)
    frame = cv2.imread(f.name+".png")
    f.close()

    blob = cv2.dnn.blobFromImage(
        frame, 1.0 / 255, None, (123.675, 116.28, 103.53), swapRB=True, crop=False)
    # First dimension is batch dimension
    blob[0][0] = blob[0][0] / 0.229
    blob[0][1] = blob[0][1] / 0.224
    blob[0][2] = blob[0][2] / 0.225

    net.setInput(blob)
    preds = net.forward()
    pred = np.array(preds)[0]
    sm = softmax(pred)

    ma = np.argmax(softmax(pred))
    score = float(sm[ma])
    return (ma, score)


def train(filepaths_train, filepaths_test, epochs=10, minibatch_size=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = ImageDataset(filepaths_train)
    test_dataset = ImageDataset(filepaths_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=minibatch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=minibatch_size, shuffle=True)

    num_classes = len(filepaths_train)
    feature_extract = True
    img = train_dataset.__getitem__(0)[0].shape

    model = models.resnet18(pretrained=True)

    # If feature_extract == False, model parameters are frozen
    set_parameter_requires_grad(model, feature_extract)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    params_to_update = model.parameters()

    criterion = CrossEntropyLoss()
    optimizer = Adam(params_to_update, lr=0.001)
    model = model.to(device)

    print("Training started")

    training_loss = []
    validation_loss = []

    training_accuracy = []
    validation_accuracy = []
    model.train()

    for epoch in range(epochs):
        train_loss = []

        for batch_index, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            curr_loss = loss.data.item()
            train_loss.append(curr_loss)

        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            correct = correct.data.item()
        train_accuracy = correct/total

        correct = 0
        total = 0
        val_loss = []
        model.eval()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss.append(loss.data.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            correct = correct.data.item()
        val_accuracy = correct/total

        training_loss.append(np.mean(np.array(train_loss)))
        training_accuracy.append(train_accuracy)

        validation_loss.append(np.mean(np.array(val_loss)))
        validation_accuracy.append(val_accuracy)

        print('Epoch : %d/%d - loss: %.4f - accuracy: %.4f - val_loss: %.4f - val_accuracy: %.4f' %
              (epoch+1, epochs, np.mean(np.array(train_loss)), train_accuracy, np.mean(np.array(val_loss)), val_accuracy))

    epoch_count = range(0, epochs)

    # torch.save(model, "audio_model.pt")

    num_channels = img[0]
    img_height = img[1]
    img_width = img[2]
    dummy_input = torch.randn(1, num_channels, img_height, img_width)
    dummy_input = dummy_input.to(device)
    torch.onnx.export(model, dummy_input, "audio_model.onnx")

    plt.subplot(121)
    plt.plot(epoch_count, training_accuracy, 'b-')
    plt.plot(epoch_count, validation_accuracy, 'r-')
    plt.legend(['Training accuracy', 'Validation accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(122)
    plt.plot(epoch_count, training_loss, 'b-')
    plt.plot(epoch_count, validation_loss, 'r-')
    plt.legend(['Training loss', 'Validation loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.show()
    print("Finished training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train an audio classification model')

    s_parser = parser.add_subparsers(
        help='Convert, train and test an audio classifier', dest="command")
    convert_parser = s_parser.add_parser('convert')
    convert_parser.add_argument(
        '-i', '--input', type=str, help="A file path to a directory that contain WAV files", required=True)
    convert_parser.add_argument(
        '-o', '--output', type=str, help="A file path to a directory where the image files are placed", required=True)

    train_parser = s_parser.add_parser('train')
    train_parser.add_argument(
        '-i', '--inputpaths', nargs='+', help="A list of path to directories in which spectrograms for training the model are placed", required=True)
    train_parser.add_argument(
        '-t', '--testpaths', nargs='+', help="A list of path to directories in which spectrograms for evaluating the model are placed", required=True)
    train_parser.add_argument(
        '-e', '--epochs', type=int, help="Number of epochs the model should be trained", required=False)
    train_parser.add_argument(
        '-m', '--minibatch', type=int, help="Size of images taken per mini batch", required=False)

    predict_parser = s_parser.add_parser('predict')
    predict_parser.add_argument(
        '-i', '--input', type=str, help="Path to the WAV file that should be classified", required=True)
    predict_parser.add_argument(
        '-m', '--modelfile', type=str, help="Path to the ONNX model file that should be used for classifying", required=True)
    args = parser.parse_args()

    if args.command == "convert":
        input_directory = args.input
        output_directory = args.output
        for f in glob.glob(os.path.join(input_directory, '*'), recursive=True):
            fbase = Path(f).stem
            save_as_spectrogram(f, os.path.join(
                output_directory, fbase+".png"))
    elif args.command == "train":
        train_datasets = args.inputpaths
        test_datasets = args.testpaths
        if args.epochs is not None:
            epochs = args.epochs
        else:
            epochs = 10

        if args.minibatch is not None:
            minibatch_size = args.minibatch
        else:
            minibatch_size = 10
        train(train_datasets, test_datasets, epochs, minibatch_size)
    elif args.command == "predict":
        audio_file = args.input
        model_onnx_path = args.modelfile
        ma, score = predict(audio_file, model_onnx_path)
        print(f"Predicated class {ma} with score {score}")
