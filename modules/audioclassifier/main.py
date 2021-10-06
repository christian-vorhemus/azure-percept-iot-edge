from azure.iot.device.iothub.aio.async_clients import IoTHubDeviceClient
from azure.iot.percept import AudioDevice
import time
import os
import sys
import asyncio
from six.moves import input
import json
import threading
from azure.storage.blob import BlobClient, BlobServiceClient
from azure.core.exceptions import AzureError
from azure.iot.device.aio import IoTHubModuleClient
from azure.iot.device.iothub.models import Message
import matplotlib.pyplot as plt
import threading
import numpy as np
import cv2
import librosa
import uuid
from shutil import copyfile

running = True
audio = AudioDevice()
cooldown = 0


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


async def send_to_iot_hub(module_client: IoTHubModuleClient, message: dict):
    await module_client.connect()
    message = Message(json.dumps(message))
    await module_client.send_message_to_output(message, "output")
    await module_client.disconnect()


async def upload_blob(device_client: IoTHubDeviceClient, audio_file: str):
    storage_info = await device_client.get_storage_info_for_blob(
        audio_file)
    success, result = store_blob(storage_info, f"./{audio_file}")
    os.remove(f"./{audio_file}")

    if success == False:
        print(f"Uploading blob failed: {str(result)}")


def store_blob(blob_info, file_name):
    try:
        sas_url = "https://{}/{}/{}{}".format(
            blob_info["hostName"],
            blob_info["containerName"],
            blob_info["blobName"],
            blob_info["sasToken"]
        )

        print("\nUploading file: {} to Azure Storage as blob: {} in container {}\n".format(
            file_name, blob_info["blobName"], blob_info["containerName"]))

        # Upload the specified file
        with BlobClient.from_blob_url(sas_url) as blob_client:
            with open(file_name, "rb") as f:
                result = blob_client.upload_blob(f, overwrite=True)
                return (True, result)

    except FileNotFoundError as ex:
        # catch file not found and add an HTTP status code to return in notification to IoT Hub
        ex.status_code = 404
        return (False, ex)

    except AzureError as ex:
        # catch Azure errors that might result from the upload operation
        return (False, ex)


def save_as_spectrogram(input_audio_file, output_image_path):
    y, sr = librosa.load(input_audio_file, sr=16000, mono=False)
    fast_fourier_signal = np.abs(librosa.stft(y[0], n_fft=255, hop_length=200))
    fast_fourier_signal_log = librosa.amplitude_to_db(
        fast_fourier_signal, top_db=None)

    dpi = 300
    height = 306
    width = 1920
    depth = 4

    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(fast_fourier_signal_log, aspect='auto', cmap='hot')
    fig.savefig(output_image_path, dpi=dpi)
    plt.close()


def predict(audio_file, image_name):
    save_as_spectrogram(audio_file, f"./{image_name}")
    net = cv2.dnn.readNetFromONNX("./audio_model.onnx")
    frame = cv2.imread(f"./{image_name}")
    # The means (0.485, 0.456, 0.406) * 255 are substracted from image, but no standard deviation division performed in blobFromImage
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

    ma = np.argmax(softmax(pred))  # if ma = 0, the sound is normal. 1 = broken
    score = 0
    if ma == 0:
        score = float(sm[0])
    else:
        score = float(sm[1])
    return(ma, score)


def recording(module_client: IoTHubModuleClient, device_client: IoTHubDeviceClient):
    global audio
    global cooldown
    while True:
        if audio.is_ready() is True:
            break
        else:
            time.sleep(1)
    while running:
        file_id = str(uuid.uuid4())
        audio_file = file_id + ".wav"
        image_file = file_id + ".png"
        print(f"Recording {audio_file}...")
        audio.start_recording(f"./{audio_file}")
        time.sleep(10)
        audio.stop_recording()
        print("Recording stopped")
        classification, score = predict(f"./{audio_file}", image_file)
        print(f"Class: {classification}, score: {score}")
        entered = False
        if classification == 1 and score >= 0.6:
            entered = True
            if cooldown <= 0:
                cooldown = 20
                msg = {"type": "faulty_valve",
                       "message": "A potentially faulty valve was detected", "score": score, "file": audio_file}
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(upload_blob(device_client, audio_file))
                loop.run_until_complete(send_to_iot_hub(module_client, msg))
                loop.close()
        time.sleep(15)
        if entered is False:
            os.remove(f"./{audio_file}")
        os.remove(f"./{image_file}")
        cooldown -= 1
        time.sleep(15)


async def main():
    try:
        global running
        if not sys.version >= "3.5.3":
            raise Exception(
                "The sample requires python 3.5.3+. Current version of Python: %s" % sys.version)
        print("IoT Hub Client for Python")

        module_client = IoTHubModuleClient.create_from_edge_environment()

        if "EdgeHubConnectionString" not in os.environ:
            raise Exception(
                "Environment variable 'EdgeHubConnectionString' is missing")

        device_client = IoTHubDeviceClient.create_from_connection_string(
            os.environ["EdgeHubConnectionString"])

        rec = threading.Thread(target=recording, args=(
            module_client, device_client,))
        rec.start()

        def stdin_listener():
            while True:
                try:
                    selection = input("Press Q to quit\n")
                    if selection == "Q" or selection == "q":
                        print("Quitting...")
                        break
                except:
                    time.sleep(10)

        print("Started")

        # Run the stdin listener in the event loop
        loop = asyncio.get_event_loop()
        user_finished = loop.run_in_executor(None, stdin_listener)

        # Wait for user to indicate they are done listening for messages
        await user_finished

        # Cancel listening
        # listeners.cancel()
        audio.close()
        running = False

        # Finally, disconnect
        await module_client.disconnect()

    except Exception as e:
        print("Unexpected error %s " % e)
        raise

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
