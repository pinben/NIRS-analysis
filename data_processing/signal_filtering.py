from scipy import signal
import numpy as np
import pywt

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    應用帶通濾波器到輸入信號。

    參數:
    - data (array-like): 輸入信號數據。
    - lowcut (float): 低切頻率(Hz)。
    - highcut (float): 高切頻率(Hz)。
    - fs (float): 採樣頻率(Hz)。
    - order (int, 可選): 濾波器的階數，默認為5。

    返回:
    array-like: 經過帶通濾波後的信號。

    說明:
    使用 Butterworth 濾波器設計帶通濾波器，然後使用前向-後向濾波應用於信號。
    """
    nyq = 0.5 * fs # Nyquist 頻率是採樣頻率的一半，用於後續的頻率正規化。
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band') # 將lowcut和highcut頻率除以 Nyquist 頻率進行正規化，轉換為 0 到 1 之間的值。
    return signal.filtfilt(b, a, data)

def wavelet_denoise(data, wavelet='db4', level=3):
    """
    使用小波變換對信號進行去噪。

    參數:
    - data (numpy.ndarray): 輸入信號數據。
    - wavelet (str, 可選): 使用的小波類型，默認為'db4'。
    - level (int, 可選): 分解的層數，默認為3。

    返回:
    array-like: 經過小波去噪後的信號。

    說明:
    使用離散小波變換進行信號分解，應用軟閾值去噪，然後重構信號。
    閾值基於最後一層小波係數的中位數絕對偏差(MAD)計算。
    """
    coeff = pywt.wavedec(data, wavelet, mode="per", level=level)
    sigma = (1/0.6745) * np.median(np.abs(coeff[-1] - np.median(coeff[-1])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeff[1:]]
    return pywt.waverec(coeff, wavelet, mode='per')

def process_signal(data, fs=10, filter=True, denoise=False, level=3, order=5, lowcut=0.01, highcut=0.5):
    """
    對輸入信號進行完整的處理，包括帶通濾波和小波去噪。

    參數:
    - data (pandas.DataFrame): 輸入信號數據，形狀為 (channels, time_points)
    - fs (float, 可選): 採樣頻率(Hz)，默認為10 Hz。
    - filter (bool, 可選): 是否進行帶通濾波，默認為True。
    - denoise (bool, 可選): 是否進行小波去噪，默認為True。
    - level (int, 可選): 小波分解的層數，默認為3。
    - order (int, 可選): 濾波器階數，默認為5。
    - lowcut (float, 可選): 帶通濾波器的低切頻率，默認為0.01 Hz。
    - highcut (float, 可選): 帶通濾波器的高切頻率，默認為0.5 Hz。

    返回:
    numpy.ndarray: 處理後的信號，形狀與輸入相同

    說明:
    1. 如果 filter=True，應用帶通濾波器(0.01 Hz 到 0.5 Hz)
    2. 如果 denoise=True，進行小波去噪
    3. 確保輸出信號長度與輸入信號相同
    """
    channels, time_points = data.shape
    processed_data = np.zeros_like(data) # 創建跟data形狀一樣的矩陣，其中元素為0

    for i in range(channels):
        channel_data = data.iloc[i].values
        # 應用帶通濾波
        if filter:
            channel_data = bandpass_filter(channel_data, lowcut, highcut, fs, order=order)
        # 應用小波去噪
        if denoise:
            channel_data = wavelet_denoise(channel_data, level=level)
            
        processed_data[i, :] = channel_data[:time_points]
    return processed_data