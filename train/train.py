import os 
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras 

#Paths
BASE_DIR =os.path.join(os.path.dirname(__file__), '..')
FEATURES_CSV = os.path.join(BASE_DIR, 'features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'audio_threat.keras')
TFLITE_PATH = os.path.join(MODEL_DIR, 'audio_threat.tflite')
LABEL_PATH = os.path.join(MODEL_DIR, 'label_map.json')

os.makedirs(MODEL_DIR, exist_ok =True)

#Load data
print("Loading features.csv...")
df = pd.read_csv(FEATURES_CSV)
print(f"Total Samples : {len(df)}")
print(f"Columns       :{list(df.columns[:3])} ... label")

x= df.drop('label', axis=1).values.astype(np.float32)
y = df['label'].values

#Encode labels
le = LabelEncoder()
y_encoded=le.fit_transform(y)
print(f"\nLabel Mapping:")
for cls, idx in zip(le.classes_, le.transform(le.classes_)):
    count= np.sum(y==cls)
    print(f"   {idx} = {cls:<15} ({count} samples)")

#Train- val -test split, 70-15-15
x_train, x_test, y_train, y_test = train_test_split(x,y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
x_train, x_val , y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

print(f"\nSplit sizes:")
print(f"  Train : {len(x_train)} samples")
print(f"  Val   : {len(x_val)} samples")
print(f"  Test  : {len(x_test)} samples")

#Build model
#3-layer dense network
# Layer 1: 128 neurons - learns broad patterns in the MFCCs
# Layer 2: 64 neurons - refines those patterns
#Layer 3:   4 neurons - one per class, outputs probabilities via softmax
model = keras.Sequential([
    keras.layers.Input(shape=(40,)),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(64,activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(len(le.classes_), activation = 'softmax')], name="audio_threat_classifier")

model.compile(
    optimizer ='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

#Train model
print("\nTraining model...")
early_stop = keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience =10,
    restore_best_weights=True,
    verbose =1
)

history= model.fit(
    x_train, y_train,
    epochs =100,
    batch_size =32,
    validation_data =(x_val, y_val),
    callbacks= [early_stop],
    verbose=1
)

#Evaluate on test set
#Run the model on the test set it has never senn during training
# This gives the most unbiased estimate of how well the model will perform on new, unseen data in the real world.
print("\nTest Set Results")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy  :{test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"Test Loss      :{test_loss:.4f}")
y_pred =np.argmax(model.predict(x_test, verbose=0), axis=1)

print("\nClassification Report:")
class_names =['alarm', 'background', 'glass_break', 'gunshot']
print(classification_report(y_test, y_pred, target_names=class_names))

print("Confusion Matrix (rows= actual, cols=predicted):")
cm= confusion_matrix(y_test, y_pred)
print(f"   Labels: {class_names}")
print(cm)

#Save the Keras Model
model.save(MODEL_PATH)
print(f"\n Keras Model saved to {MODEL_PATH}")

#Convert to TFLite
print ("\nConverting to TFLite  with INT8 quantization...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for sample in x_train[:200]:
        yield[sample.reshape(1, -1).astype(np.float32)]

converter.representative_dataset= representative_data_gen
tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

keras_size = os.path.getsize(MODEL_PATH)/ 1024
tflite_size = os.path.getsize(TFLITE_PATH)/ 1024
print(f"TFLite model saved to {TFLITE_PATH}")
print(f"Model Size: {keras_size:.1f} KB (keras) -> {tflite_size:.1f} KB (TFLite)")

#Save label map
label_map={
    str(i): class_names[i]
    for i in range(len(class_names))
    }
with open(LABEL_PATH, 'w') as f:
    json.dump(label_map, f, indent=2)
print(f"Label Map saved to {LABEL_PATH}")
print(f"\nLabel map: {label_map}")

print("\nTraining complete!")