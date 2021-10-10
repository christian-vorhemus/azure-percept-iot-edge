This is a sample Azure IoT Edge module that runs an audio classification predictive maintenance model on the Azure Percept device and sends messages to Azure IoT Hub if a damaged machine is detected. README under construction.

## Train an audio classifier

### Install and prepare
1. Make sure that you have Git, Python 3 (3.8 recommended) and Docker installed on your machine.
2. Clone this repository `git clone https://github.com/christian-vorhemus/azure-percept-iot-edge.git`. If you are using Azure DevOps, use `git clone https://dev.azure.com/chrysalis-innersource/Azure%20Percept%20Python%20library/_git/Azure%20Percept%20IoT%20Edge%20Module`
3. Change directory into the modules/audioclassifier of the just cloned repository and clone the Azure Percept Python library `git clone https://github.com/christian-vorhemus/azure-percept-py.git` into it. The folder "azure-percept-py" should now be present within modules/audioclassifier.
4. Change directory into the `ml` folder and run `pip install -r requirements.txt`. If you see errors please investigate them for more information. For example, some packages might not exist for certain platforms.

### Train and test
1. Prepare your audio datatset you want to use. If you don't have one yet you can use the [Malfunctioning Industrial Machine Investigation](https://zenodo.org/record/3384388#.YWKiqflBzOi) dataset. Make sure that you have at least 2 sets, a directory containing several WAV files of **working** parts and a directory of WAV files containing **damaged** parts.
2. Convert the WAV files into spectrograms using `python ml.py convert --input /path/to/wav/files --output /output/path/images`. Do this for every directory you have.
3. Use a script or manually divide the images into a training and a test set. Usually 80% of images per class are used for training, the rest for test. If you have 2 classes (damaged and intact), you should have 4 folders: 2 folders for the training data for damaged and intact parts, 2 directories for the test data for damaged and intact parts.
4. Now train a machine learning classifier pointing to the training folders of images containing the spectrograms as input (`-i`) as well as the test folders (`-t`) `python ml.py train -i /output/path/images/damaged/training /output/path/images/intact/training -t /output/path/images/damaged/test /output/path/images/intact/test -e 10`
5. After training, a new file `audio_model.onnx` should be present in your `ml/` working directory. You can use this model to now make predictions for single audo files, for example `python ml.py predict -i /path/to/file.wav -m /path/to/audio_model.onnx`. 
6. Make sure that you move `audio_model.onnx` into the modules/audioclassifier directory.

## Build and push Docker Image
1. Make sure you set the environment variable `EdgeHubConnectionString` in `deployment.template.json` to the connection string of your IoT Hub. The module uses it to send telemetry messages to the IoT Hub.
2. Change the `repository` property in `module.json` to your name of the container registry you use.
3. To build an Azure IoT Edge module, you can follow the [official documentation](https://docs.microsoft.com/en-us/azure/iot-edge/tutorial-python-module?view=iotedge-2020-11#build-and-push-your-module). To build a docker image, change into the modules/audioclassifier directory and run `docker build -f Dockerfile.arm64v8 -t <YOURNAME>.azurecr.io/audioclassifier:0.0.1-arm64v8 .` assuming you have an Azure Container Registry called `<YOURNAME>`. Afterwards push it to your registry using `docker push <YOURNAME>.azurecr.io/audioclassifier:0.0.1-arm64v8`

## Deploy IoT Edge Module to Azure Percept
If you want to use the Azure Portal, please refer to [this guide](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-deploy-modules-portal?view=iotedge-2020-11) how to run images from a container registry on an Azure IoT Edge device.
