# Acoustic Predictive Maintenance with Azure Percept

This repository contains the open sourced part of the acoustic predictive maintenance project with Azure Percept where defective machines are detected by their sound. The classification is done locally using a melspectrogram image that is evaluated on the hardware accelerated VPU (Visual Processing Unit) of Azure Percept. The software is written in Python and hosted as an Azure IoT Edge Module.

## Background and architecture

Many industrial machines make noises which can be used to detect whether the machine is faulty or could soon be defective. Detecting and classifying these sounds can be performed either manually or automatically, whereby the requirement is usually that automated classification must be performed locally ("on the edge") in order to ensure a self-sufficient environment that also functions in the event of temporary Internet failures. Processing in the cloud is particularly useful if further activities are to be carried out, such as sending a notification to a technician as soon as a potentially defective machine has been detected. The approach described here is therefore a hybrid architecture in which the audio recording and the assessment of whether the noise from the machine indicates a fault is performed locally; if this is the case, a message is sent to a message broker in the cloud. The architecture below describes the process in more detail. 

## Train an audio classifier

### Install and prepare
1. Make sure that you have Git, Python 3 (3.8 recommended) and Docker installed on your machine.
2. Clone this repository `git clone https://github.com/christian-vorhemus/azure-percept-iot-edge.git`.
3. Change directory into the `ml` folder and run `pip install -r requirements.txt`. If you see errors please investigate them for more information. For example, some packages might not exist for certain platforms.

### Train an audio classifier machine learning model
1. Prepare your audio dataset you want to use. If you don't have one yet you can use the [Malfunctioning Industrial Machine Investigation](https://zenodo.org/record/3384388#.YWKiqflBzOi) dataset. Make sure that you have at least 2 sets, a directory containing several WAV files of **working** parts and a directory of WAV files containing **damaged** parts.
2. Convert the WAV files into spectrograms using `python ml.py convert --input /path/to/wav/files --output /output/path/images`. Do this for every directory you have.
3. Use a script or manually divide the images into a training and a test set. Usually, 70-80% of images per class are used for training, the rest for test. If you have 2 classes (damaged and intact), you should have 4 folders: 2 folders for the training data for damaged and intact parts, 2 directories for the test data for damaged and intact parts.
4. Now train a machine learning classifier pointing to the training folders of images containing the spectrograms as input (`-i`) as well as the test folders (`-t`) `python ml.py train -i /output/path/images/intact/training /output/path/images/damaged/training -t /output/path/images/intact/test /output/path/images/damaged/test -e 10`
5. After training, a new file `audio_model.onnx` should be present in your `ml/` working directory. You can use this model to now make predictions for single audo files, for example `python ml.py predict -i /path/to/file.wav -m /path/to/audio_model.onnx`. 
6. Make sure that you move `audio_model.onnx` into the modules/audioclassifier directory.

## Build and push Docker image
1. Make sure you set the environment variable `EdgeHubConnectionString` in `deployment.template.json` to the connection string of your IoT Hub. The module uses it to send telemetry messages to the IoT Hub.
2. Change the `repository` property in `module.json` to your name of the container registry you use.
3. You may need to change the integer value in `main.py` that describes if a part is faulty or okay. The if-statement `if classification == 1 and score >= 0.6:` assumes that `audio_model.onnx` returns 1 whenever a damaged part was detected. Depending on how you trained the model and how many classes you have this value might be different.
4. Make sure that you [connect a storage account to your IoT Hub](https://docs.microsoft.com/en-us/azure/iot-hub/iot-hub-python-python-file-upload#associate-an-azure-storage-account-to-iot-hub). The module uses this storage account to upload .WAV files.
5. To build an Azure IoT Edge module, you can follow the [official documentation](https://docs.microsoft.com/en-us/azure/iot-edge/tutorial-python-module?view=iotedge-2020-11#build-and-push-your-module). To build a docker image, change into the modules/audioclassifier directory and run `docker build -f Dockerfile.arm64v8 -t <YOURNAME>.azurecr.io/audioclassifier:0.0.1-arm64v8 .` assuming you have an Azure Container Registry called `<YOURNAME>`. Afterwards push it to your registry using `docker push <YOURNAME>.azurecr.io/audioclassifier:0.0.1-arm64v8`

## Deploy IoT Edge Module to Azure Percept
If you want to use the Azure Portal, please refer to [this guide](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-deploy-modules-portal?view=iotedge-2020-11) how to run images from a container registry on an Azure IoT Edge device.
