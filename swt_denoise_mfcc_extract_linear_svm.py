""""
Here is my test for denoising a aoudio signal with swt, extract the mfccs and then train a linear svm 
to detect and classify the whale species.
I tested with low data because the database is currently under maintanace and not accessable.
"""


# Core imports
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pywt
from scipy.signal.windows import tukey
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib  # for saving model + scaler

# configuration variables
base_dir = "C:/Users/luede/Seafile/WhaleData" #direction of species folders
sr = 16384   #sampling rate to use
n_mfcc = 40 #number of parameter to extract per frame
n_fft = 1024 #fft size for mfcc extraction (larger-more frequency resolution, less time resolution) windiow length is n_fft/sr
hop_length = 256 #hop length for mfcc extraction (large means less overlap ) common is hop_length = n_fft // 4
n_mels = 40 #number of mel bands to use for mfcc extraction (probably a lot for whales because high frerquencie band, just test with low like 40, usefull more like 128 upwards)


denoise_method = 'swt' #method for denoising (swt, spectral gating, none)
wavelet = 'db4' #wavelet type for swt denoising
level = 7 #level of swt denoising
threshold = 0.03 #threshold for swt denoising
mode = 'soft' #mode of thresholduing (soft, hard)
species_list = ['HumpbackWhale', 'CommonDolphin', "SouthernRightWhale"] #list of species to process (unfortunatly the databased is closed atm so only this)
percentile = 70 #percentile for percentile based denoising
#feats and labels for training 
X_feats = []
y_labels = []
file_paths = []
k_factors = None #factor for level based thresholding, none is default decreasing with level (because lower frequencies have more energy usually)


#windowing parameters rto classify long signals
window_seconds = 5.0    # length of each classification window
window_overlap = 0.5    # 50% overlap
tukey_alpha = 0.5       # Tukey window parameter
min_prob_whalle = 0.8    # minimum probability to consider a window as containing whale vocalization



###########PAddding functions############

#padding so that for swt denoising the length is multiple of 2**level becuase you can only use if the length a multiple of 2**level
def zero_pad(x, level):
    N = len(x)
    divisor = 2 ** level
    pad_amount = (-N) % divisor
    if pad_amount > 0:
        x = np.concatenate([x, np.zeros(pad_amount)])
    return x




#################Stationary wavelet transform denoising functions ############

#Here are some sort of different swt denoising functions because im currently testing what denoises the best and what results do work well


#basic swt denosie function
#threshold and mode are preset for soft and for the visushrink gaussion noice standard deviation threshold for testing
def swt_denoise(x, level, wavelet, threshold, mode):
    x=zero_pad(x, level)
    coeffs = pywt.swt(x, wavelet, level=level)
    denoised_coeffs = []
    for (cA, cD) in coeffs:
        #calculate the standrard deviation of noice with assuming gaussion noice
        sigma= np.median(np.abs(cD)) / 0.6745
        #calculate the thrshold with visushrink rule
        threshold = sigma * np.sqrt(2 * np.log(len(x)))
        #soft thresholding
        cD_denoised = np.sign(cD) * np.maximum(np.abs(cD) - threshold, 0.0)
        denoised_coeffs.append((cA, cD_denoised))
    return pywt.iswt(denoised_coeffs, wavelet, level)


#basic swt denosie function with pywt
def swt_denoise_pywt(x, level, wavelet, threshold, mode):
    x=zero_pad(x, level)
    coeffs = pywt.swt(x, wavelet, level=level)
    denoised_coeffs = []
    for (cA, cD) in coeffs:
        cD_denoised = pywt.threshold(cD, threshold, mode=mode)
        denoised_coeffs.append((cA, cD_denoised))
    return pywt.iswt(denoised_coeffs, wavelet, level)

##############currently working on level baes####################
#extract coeffs with swt, threshold and compute signal back with iswt but for every level the same threshold
def swt_denoise_level_based(x, wavelet, level, threshold, k_factors, mode):
    #padding
    x=zero_pad(x, level)
    coeffs = pywt.swt(x, wavelet, level=level)
    denoised_coeffs = []
    for j, (cA, cD) in enumerate(coeffs):
        cD_denoised = pywt.threshold(cD, threshold, mode=mode)
        denoised_coeffs.append((cA, cD_denoised))
    return pywt.iswt(denoised_coeffs, wavelet)


#level based percentile thresholding
def denoise_signal_percentile (signal, wavelet, level, percentile):
  coeffs = pywt.swt(signal, wavelet, level=level)
  new_coeffs = []
  for j, (cA_j, cD_j) in enumerate(coeffs, start=1):
        
    #tired a gp kernel model for level dependent thresholding but takes waay too long
    thr = np.percentile(np.abs(cD_j), 60)
    # basic level deppendent soft thresholding
    cD_new = np.sign(cD_j) * np.maximum(np.abs(cD_j) - thr, 0.0)
    new_coeffs.append((cA_j, cD_new))
  return pywt.iswt(new_coeffs, wavelet, level)




################mfcc extraction functions##############


#extract the mfccs and project them onto a 1 dim vector for linear svm
##############trying to optimaize the projection, not happy with that yet
def extract_mfcc_features(x, sr, n_mfcc, n_fft, hop_length, n_mels):
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    feat = np.concatenate([mfcc_mean, mfcc_std])
    return feat





################Test and training the svm####################


if __name__ == "__main__":
    #########################variables#########################
    base_dir = "C:/Users/luede/Seafile/WhaleData"
    target_sr=16384
    n_mfcc=40
    use_denoise = True
    wavelet = 'db4'
    level = 7
    threshold = 0.03
    mode = 'soft'
    species_list = ['HumpbackWhale', 'CommonDolphin',"SouthernRightWhale"]
    X_feats = []
    y_labels = []
    file_paths = []



    for label, sp in enumerate(species_list):
        folder = os.path.join(base_dir, sp)
        if not os.path.isdir(folder):
            print(f"WARNING: folder not found: {folder}")
            continue
        for filename in os.listdir(folder):
                if not filename.lower().endswith((".wav", ".flac", ".aiff", ".aif")):
                    continue

                path = os.path.join(folder, filename)
                try:
                    x, sr = librosa.load(path, sr=target_sr, mono=True)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
                if use_denoise:
                    x = zero_pad(x, level)
                    x = swt_denoise_pywt(x, level, wavelet, threshold, mode)
                mfccs = extract_mfcc_features(x, sr, n_mfcc, n_fft, hop_length, n_mels)
                X_feats.append(mfccs)
                y_labels.append(label)
                file_paths.append(path)
    print("MFCC extraction complete.")

    # daten normalisieren x sind features, y sind labels 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feats)


    # split data for train and test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_labels, test_size=0.2, random_state=42, stratify=y_labels)


    # train svm
    svm_clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    svm_clf.fit(X_train, y_train)
    print("SVM training complete.")

    ###can use this if you want to show the results
    y_pred = svm_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", acc)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=species_list))








################Here a test to classify a longer signal, its a mixture from noises from the arctis which a very close to whale calls from frequency and time##########################
#doesnt work very good but its mainly for testing the whole pipeline
# remove the """ and """ to run this part
"""

test_file = "C:/Users/luede/Seafile/WhaleData/Teststreams/humpbackdolphtest.wav"
x_long, sr_long = librosa.load(test_file, sr=target_sr, mono=True)

window_sec = 5.0
overlap = 0.5
min_prob = 0.8
tukey_alpha = 0.5
use_denoise = True

# use sr_long here, not sr
win_len = int(window_sec * sr_long)
hop_len = int(win_len * (1.0 - overlap))
if hop_len <= 0:
    hop_len = win_len

# pad x_long if it is shorter than one window
if len(x_long) < win_len:
    x_long = np.pad(x_long, (0, win_len - len(x_long)))

# framing on x_long, not x
frames = librosa.util.frame(x_long, frame_length=win_len, hop_length=hop_len)
n_frames = frames.shape[1]

window = tukey(win_len, alpha=tukey_alpha)

times = []
preds = []
probs = []

for i in range(n_frames):
    frame = frames[:, i].astype(np.float32)
    frame = frame * window

    if use_denoise:
        frame = swt_denoise_pywt(frame, wavelet=wavelet, level=level, threshold=threshold, mode=mode)

    feat = extract_mfcc_features(frame, sr_long, n_mfcc, n_fft, hop_length, n_mels)
    feat_scaled = scaler.transform(feat.reshape(1, -1))

    prob_vec = svm_clf.predict_proba(feat_scaled)[0]
    max_prob = np.max(prob_vec)
    pred_idx = np.argmax(prob_vec)

    if max_prob < min_prob:
        preds.append(-1)  # uncertain / no-whale
    else:
        preds.append(pred_idx)

    probs.append(max_prob)

    # time in seconds for the center of this window
    start_sample = i * hop_len
    center_sample = start_sample + win_len // 2
    times.append(center_sample / sr_long)

# now convert to numpy (outside the loop)
times = np.array(times)
preds = np.array(preds)
probs = np.array(probs)

# build a list of (time, label, prob)
results = []
for t, p, pr in zip(times, preds, probs):
    if p == -1:
        label = "uncertain/no-whale"
    else:
        label = species_list[p]
    results.append((t, label, pr))

# print the list
for t, label, pr in results:
    print(f"{t:8.2f} s  ->  {label:20s}  (p = {pr:.2f})")
"""