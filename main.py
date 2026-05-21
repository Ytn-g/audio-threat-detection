import os
import sys
import json 
import time 
import threading
import queue
import numpy as np
import pyaudio
import cv2
import paho.mqtt.client as mqtt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from capture import extract_from_array

#Config(Settings you can tune without touching the rest of the code)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'audio_threat.tflite')
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), 'model', 'label_mpap.json')

#audio settings
RATE = 16000
CHUNK = 1024
DURATION = 1.0
#Camera Settings
CAMERA_INDEX =0
MOTION_THRESH = 8.0
#Inference settings
CONFIDENCE_THRESH = 0.75
FUSION_WINDOW =3.0
#MQTT settings
MQTT_BROKER ='localhost'
MQTT_PORT = 1883
MQTT_TOPIC = 'threats/corroborated'

#LOAD MODEL
print("Loading TFLite model...")
interpreter =tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Model loaded. Input shape: {input_details[0]['shaope']}")

#LOAD LABEL MAP
with open(LABEL_MAP_PATH) as f:
    label_map =json.load(f)
print(f"Labels: {label_map}")

#SHARED STATE
motion_timestamps = []
motion_lock = threading.Lock()
audio_queue = queue.Queue(maxsize=10)
running = threading.Event()
running.set()

#MQTT SETUP
mqtt_connected = False
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

def on_connect(client, userdata, flags, reason_code, properties):
    global mqtt_connected
    if reason_code == 0:
        mqtt_connected = True
        print("Connected to MQTT broker")
    else:
        print(f"MQTT connection failed: {reason_code}")

mqtt_client.on_connect = on_connect

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    mqtt_client.loop_start()
    time.sleep(0.5)
except Exception as e:
    print(f"MQTT broker not available ({e}) - alerts will print to console only")

#MOTION DETECTION THREAD
def motion_thread():
    print("Starting motion detection (camera index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Warning: could not open camera - motion detection disabled")
        print("Alerts will fire on audio only( no fusion)")
        return
    
    ret, prev_frame = cap.read()
    if not ret:
        print("WARNING: Could not read from camera")
        cap.release()
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    print("Motion detection running")

    while running.is_set():
        ret, frame =cap.read()
        if not ret:
            continue
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff= cv2.absdiff(prev_gray, gray)
        motion_score =diff.mean()

        if motion_score > MOTION_THRESH:
            with motion_lock:
                motion_timestamps.append(time.time())
                #keep only the last 20 timestamps to save memory
                if len(motion_timestamps) > 20:
                    motion_timestamps.pop(0)
        prev_gray = gray
        time.sleep(0.05)
    cap.release()
    print("Motion thread stopped")
# AUDIO Capture Thread
def audio_capture_thread():
    print("Starting audio capture...")
    p = pyaudio.PyAudio()

    print("Available audio input devices:")
    for i in range (p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"[{i}] {info['name']}")
    
    try:
        stream = p.open(
            format =pyaudio.paInt16,
            channels=1,
            rate= RATE,
            input= True,
            frames_per_buffer = CHUNK
        )
    except Exception as e:
        print(f"ERROR:Could not open microphone: {e}")
        running.clear()
        return
    
    print("Audio capture running- listening for threats...")
    n_chunks = int(RATE / CHUNK * DURATION)

    while running.is_set():
        frames = []
        for _ in range (n_chunks):
            if not running.is_set():
                break
            try:
                data = stream.read(CHUNK, exception_on_overflow = False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            except Exception as e:
                print(f"Audio capture error: {e}")
                continue 
        if frames:
            #Combine chunks into one 1-second array and normalize to [-1, 1]
            audio_array = np.concatenate(frames).astype(np.float32) /32768.0
            try:
                audio_queue.put_nowait(audio_array)
            except queue.Full:
                pass
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio capture stopped")

#Inference Thread
def inference_thread():
    print("Inference thread running...")
    alert_count=0

    while running.is_set():
        try:
            audio_array = audio_queue.get(timeout = 1.0)
        except queue.Empty:
            continue

        #Extract the MFCC features from the audio array
        try:
            features = extract_from_array(audio_array)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            continue
        #Run the TFLite inference
        input_data = features.reshape(1, -1).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        probabilities = interpreter.get_tensor(output_details[0]['index']) [0]

        #get the top prediction and confidence
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        predicted_label = label_map[str(predicted_idx)]

        #print live status every infernce
        bar = '█' * int(confidence * 20) + '░' * (20 - int(confidence * 20))
        print(f"\r[{bar}] {confidence:.2f} - {predicted_label: <15}", end='', flush=True)

        #Skip if its background or below confidence threshold
        if predicted_label =='background' or confidence < CONFIDENCE_THRESH:
            continue
        #fusion
        now = time.time()
        with motion_lock:
            recent_motion = any(now- t < FUSION_WINDOW for t in motion_timestamps)
        
        #alert payload
        payload= {
            'event': predicted_label,
            'confidence': round(confidence, 4),
            'timestamp': now,
            'motion': recent_motion,
            'corroborated': recent_motion
        }
        if recent_motion:
            alert_count +=1
            print(f"\n🚨 CORROBORATED ALERT#{alert_count}: {predicted_label.upper()}"
                  f"(confidence: {confidence:.2f}) - audio and motion detected!")
            
            #publish to MQTT if connected
            if mqtt_connected:
                mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
                mqtt_client.publish('threats/all', json.dumps(payload))
            else:
                print(f"   MQTT offline- payload: {json.dumps(payload)}")
        else:
            print(f"\n⚠️ AUDIO ONLY: {predicted_label.upper()}"
                  f"(confidence: {confidence:.2f}) - no recent motion, suppressed")
    print("Inference thread stopped")

#FUSION HELPER
def motion_recent(window=FUSION_WINDOW):
    now =time.time()
    with motion_lock:
        return any(now - t < window for t in motion_timestamps)

#MAIN
if __name__ == '__main__':
    print("\n" + "="*50)
    print("    Audio Threat Detection System Starting...")
    print("    Press Ctrl+C to stop")
    print("="*50 + "\n")

    #Starting all 3 threads
    t_motion = threading.Thread(target= motion_thread,       daemon=True)
    t_audio  = threading.Thread(target=audio_capture_thread, daemon=True)
    t_inference=threading.Thread(target=inference_thread,    daemon=True)

    t_motion.start()
    time.sleep(0.5)
    t_audio.start()
    time.sleep(0.5)
    t_inference.start()

    print("\nAll systems running. Listening for threats...\n")

    try:
        #Keep the main thread alive while the others do their work
        while running.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        running.clear()#Signal all threads to stop

    #Wait a moment for threads to finish
    t_motion.join(timeout=2)
    t_audio.join(timeout=2)
    t_inference.join(timeout=2)

    if mqtt_connected:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    print("System stopped. Goodbye!")



