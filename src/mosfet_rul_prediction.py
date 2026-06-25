# ==========================================================
# FORCE GPU
# ==========================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================================
# IMPORTS
# ==========================================================
import glob
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ==========================================================
# CONFIG
# ==========================================================
DATA_DIR = '/content/drive/
SEQ_LEN = 80
SMOOTH_WIN = 5

# ==========================================================
# GPU SETUP
# ==========================================================
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================
def extract_features(mat_file):
    try:
        mat = sio.loadmat(mat_file)
        steady = mat['measurement']['steadyState'][0,0]
        td = steady['timeDomain'][0]

        Vds, Id, Temp = [], [], []

        for t in td:
            Vds.append(t['drainSourceVoltage'][0,0])
            Id.append(t['drainCurrent'][0,0])
            Temp.append(t['packageTemperature'][0,0])

        Vds = np.array(Vds)
        Id = np.array(Id)
        Temp = np.array(Temp)

        valid = Id > 0.1
        Vds, Id, Temp = Vds[valid], Id[valid], Temp[valid]

        Rds = Vds / Id

        # smoothing
        Rds_s = np.convolve(Rds, np.ones(SMOOTH_WIN)/SMOOTH_WIN, mode='valid')
        Temp_s = np.convolve(Temp, np.ones(SMOOTH_WIN)/SMOOTH_WIN, mode='valid')

        # derivatives
        dRds = np.gradient(Rds_s)
        d2Rds = np.gradient(dRds)

        # additional features
        cycle = np.arange(len(Rds_s)) / len(Rds_s)
        log_Rds = np.log(Rds_s + 1e-8)
        power = Vds[:len(Rds_s)] * Id[:len(Rds_s)]

        return np.column_stack((Rds_s, dRds, d2Rds, Temp_s, cycle, log_Rds, power))

    except:
        return None

# ==========================================================
# LOAD DATA
# ==========================================================
files = sorted(glob.glob(DATA_DIR + "/**/*.mat", recursive=True))

data = []
for f in files:
    d = extract_features(f)
    if d is not None and len(d) > SEQ_LEN:
        data.append(d)

print("Valid devices:", len(data))

# ==========================================================
# SPLIT
# ==========================================================
n = len(data)
train = data[:int(0.6*n)]
val   = data[int(0.6*n):int(0.8*n)]
test  = data[int(0.8*n):]

# ==========================================================
# SCALING
# ==========================================================
scaler = StandardScaler()
scaler.fit(np.vstack(train))

train = [scaler.transform(x) for x in train]
val   = [scaler.transform(x) for x in val]
test  = [scaler.transform(x) for x in test]

# ==========================================================
# BUILD DATASET (FIXED RUL)
# ==========================================================
def build_dataset(dataset):
    X, y = [], []

    for d in dataset:
        n = len(d)

        for i in range(n - SEQ_LEN):
            X.append(d[i:i+SEQ_LEN])

            # per-device normalized RUL
            rul = (n - (i + SEQ_LEN)) / n
            y.append(rul)

    return np.array(X), np.array(y)

Xtr, ytr = build_dataset(train)
Xva, yva = build_dataset(val)
Xte, yte = build_dataset(test)

# convert dtype
Xtr = Xtr.astype(np.float32)
Xva = Xva.astype(np.float32)
Xte = Xte.astype(np.float32)

ytr = ytr.astype(np.float32)
yva = yva.astype(np.float32)
yte = yte.astype(np.float32)

# ==========================================================
# DATA PIPELINE
# ==========================================================
train_ds = tf.data.Dataset.from_tensor_slices((Xtr, ytr)).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((Xva, yva)).batch(128)

# ==========================================================
# PURE LSTM MODEL (OPTIMIZED)
# ==========================================================
inputs = Input(shape=(SEQ_LEN, Xtr.shape[2]))

x = LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4))(inputs)
x = Dropout(0.3)(x)

x = LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.3)(x)

x = LSTM(32, kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.2)(x)

x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)

outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss='mse'
)

model.summary()

# ==========================================================
# TRAIN
# ==========================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=120,
    callbacks=[
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(patience=7, factor=0.5)
    ],
    verbose=1
)

# ==========================================================
# PREDICTION (DEVICE LEVEL)
# ==========================================================
pred = model.predict(Xte).flatten()
actual = yte

device_preds = []
device_actual = []

idx = 0

for d in test:
    n = len(d)
    count = n - SEQ_LEN

    rul_preds = pred[idx:idx+count] * n

    est_life = []
    for i in range(count):
        pos = i + SEQ_LEN
        est_life.append(pos + rul_preds[i])

    device_preds.append(np.mean(est_life))
    device_actual.append(n)

    idx += count

device_preds = np.array(device_preds)
device_actual = np.array(device_actual)

# ==========================================================
# METRICS
# ==========================================================
r2 = r2_score(device_actual, device_preds)
mae = mean_absolute_error(device_actual, device_preds)
rmse = np.sqrt(mean_squared_error(device_actual, device_preds))

print("\nFINAL RESULTS")
print("R2:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# ==========================================================
# PLOTS
# ==========================================================

# Loss
plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.title("Training Loss")
plt.show()

# RUL curve
plt.figure()
plt.plot(actual[:300]*100, label='Actual')
plt.plot(pred[:300]*100, label='Predicted')
plt.legend()
plt.title("RUL Prediction (Normalized %)")
plt.show()

# Device-level scatter
plt.figure()
plt.scatter(device_actual, device_preds)
plt.xlabel("Actual Life")
plt.ylabel("Predicted Life")
plt.title("Device-Level Prediction")
plt.show()

# Error distribution
plt.figure()
plt.hist(device_actual - device_preds, bins=20)
plt.title("Error Distribution")
plt.show()

