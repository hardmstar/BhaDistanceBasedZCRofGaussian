# -*-coding:utf-8 -*-
'''
hardmstar@126.com
generate features
tensorflow 1.4
use librosa rather than pyaudioanalysis needing libmagic
reference
[1] pyaudioanalysis.FeatureExtraction
[2] GeMAPS for Voice Research and Affective Computing
[3] Jitter and Shimmer Measurements for Speaker Recognition
'''

from dataset import *
import librosa
import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from scipy import signal
from audiolazy.lazy_lpc import lpc

eps = 0.00000001


def genFeatures(wav_file, wav_feature_folder=None):
    # generate features including st temporal features and spectrogram features
    y, sr = librosa.load(wav_file, sr=None)  # y: audio time series , sr: sampling rate
    win = 0.03  # window: 30ms
    step = 0.010  # step : 10ms
    features = stFeatureExtraction(y, sr, win * sr, step * sr)


def stFeatureExtraction(y, sr, win, step):
    win = int(win)
    step = int(step)
    # signal normalization to [-1,1]
    y = np.double(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    N = len(y)  #

    cur_p = 0
    cnt_fr = 0
    features = []
    while (cur_p + win - 1 < N):
        cnt_fr += 1
        y_frame = y[cur_p:cur_p + win]
        cur_p += step

        features_frame = []
        # temporal features
        # features_frame.append(stZCR(y_frame))
        # features_frame.append(stEnergy(y_frame))
        # features_frame.append(stShimmerDB(y_frame))
        # features_frame.append(stShimmerRelative(y_frame))
        # features_frame.append(audioFeatureExtraction.stHarmonic(y_frame,sr))
        features_frame.append(stIntensity(y_frame))
        features.append(features_frame)
    print(features)
# frequency features


def stZCR(frame):
    # computing zero crossing rate
    count = len(frame)
    count_z = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return (np.float64(count_z) / np.float64(count - 1))


def stEnergy(frame):
    return (np.sum(frame ** 2) / np.float64(len(frame)))


def stShimmerDB(frame):
    '''
     amplitude shimmer 振幅扰动度
     expressed as variability of the peak-to-peak amplitude in decibels 分贝
     [3]
    '''
    count = len(frame)
    sigma = 0
    for i in range(count):
        if i == count - 1:
            break
        sigma += np.abs(20 * (np.log10(np.abs(frame[i + 1] / (frame[i] + eps)))))
    return np.float64(sigma) / np.float64(count - 1)


def stShimmerRelative(frame):
    '''
    shimmer relative is defined as average absolute difference between the amplitude
    of consecutive periods divided by the average amplitude, expressed as percentage
    [3]
    '''
    count = len(frame)
    sigma_diff = 0
    sigma_sum = 0
    for i in range(count):
        if i < count - 1:
            sigma_diff += np.abs(np.abs(frame[i]) - np.abs(frame[i + 1]))
        sigma_sum += np.abs(frame[i])
    return np.float64(sigma_diff / (count - 1)) / np.float64(sigma_sum / count + eps)


def stLoudness(frame):
    '''
    loudness calculation
    '''
    pass


def stHNR(frame, period):
    '''
    harmonics to noise ratio 谐噪比
    HNR = 10 * log( ACF(T0) / (ACF(0) - ACF(T0)) )
	frame: a frame of signal
	period: pitch period of given frame
	return hnr db
	'''
    period = int(period)
    if period == 0:
        return 0  # when pitch period is zero, return zero
    ru = np.correlate(frame, frame, mode='full')
    win = len(frame)
    print(np.where(ru == np.max(ru)))
    ru = ru[win - 1:]
    print(np.max(ru))
    print(ru[period])
    HNR = 10 * np.log(ru[period] / (ru[0] - ru[period]))
    return HNR


#####################################
##
## convert matlab code to python code
## pretreatment
##
######################################

def enframe(y, win, step):
    # done
    win = int(win);
    step = int(step)
    N = len(y)  # data length
    nf = int((N - win + step) / step)
    frameout = np.zeros((nf, int(win)))

    cur_p = 0
    cnt_fr = 0
    while (cur_p + win - 1 < N):
        frameout[cnt_fr, :] = y[cur_p: cur_p + win]
        cnt_fr += 1
        cur_p += step
    print(frameout)
    return frameout


def FrameTimeC(frameNum, framelen, inc, fs):
    frameTime = (np.array(list(range(frameNum))) * inc + framelen / 2) / fs
    return frameTime


####################################
##
## calculate pitch with correlation
## calPitch() JitterAbsolute() JitterRelative()
##
####################################	

class voiceSegment:

    def __init__(self, in1=0, in2=0, duratioin=0):
        self.begin = in1
        self.end = in2
        self.duratioin = duratioin


def pitch_vad(x, win, step, T1, miniL):
    # 端点检测
    y = enframe(x, win, step).T
    fn = len(y[0, :])
    print(fn)
    Esum = []  # energy of frames
    H = []  # spectrom entropy
    for i in range(fn):
        Sp = np.abs(np.fft.fft(y[:, i]))
        Sp = Sp[:int(win / 2)]  # fft positive
        Esum.append(np.sum(Sp ** 2))  # energy
        prob = Sp / (np.sum(Sp))  # probability
        H.append(- np.sum(prob * np.log(prob + eps)))

    H = np.array(H)
    hindex = np.where(H < 0.1)
    H[hindex] = np.max(H)
    # Ef = np.sqrt(1 + np.abs(Esum / np.linalg.inv(H)))  # energy entropy percentage
    Ef = np.sqrt(1 + np.abs(Esum / H))
    Ef = Ef / np.max(Ef)

    zindex = np.where(Ef >= T1)  # 寻找Ef中大于T1的部分
    zseg = findSegment(zindex)  # 给出端点检测各段的信息
    zsl = len(zseg)  # 给出段数
    j = 0
    SF = np.zeros(fn)
    voiceseg = []
    for k in range(zsl):
        if zseg[k].duratioin >= miniL:
            j = j + 1
            in1 = zseg[k].begin
            in2 = zseg[k].end
            voiceseg.append(zseg[k])
            SF[in1:in2] = 1

    vosl = len(voiceseg)  # 有话段的段数
    return voiceseg, vosl, SF, Ef


def findSegment(express):
    # express = np.array(express)
    '''
    if express[0][0] == 0:
        voiceIndex = np.where(express == 1)
    else:
        voiceIndex = express
    '''
    voiceIndex = np.array(express).flatten()
    soundSegment = []
    k = 0
    soundSegment.append(voiceSegment(voiceIndex[0]))
    for i in range(len(voiceIndex) - 1):
        if voiceIndex[i + 1] - voiceIndex[i] > 1:
            soundSegment[k].end = voiceIndex[i]
            soundSegment.append(voiceSegment(voiceIndex[i + 1]))
            k = k + 1
    soundSegment[k].end = voiceIndex[-1]

    for i in range(k + 1):
        soundSegment[i].duratioin = soundSegment[i].end - soundSegment[i].begin + 1
    return soundSegment


def pitch_Corr(x, win, step, T1, sr, miniL=10):
    win = int(win);
    step = int(step)
    vseg, vsl, SF, Ef = pitch_vad(x, win, step, T1, miniL)
    y = enframe(x, win, step).T
    fn = len(SF)
    lmin = int(sr / 500)
    lmax = int(sr / 27.5)
    period = np.zeros(fn)
    for i in range(vsl):
        ixb = vseg[i].begin
        ixe = vseg[i].end
        ixd = vseg[i].duratioin
        for k in range(ixd):
            u = y[:, k + ixb]
            ru = np.correlate(u, u, mode='full')
            ru = ru[win - 1:]  # positive
            tloc = np.array(np.where(ru[lmin:lmax] == np.max(ru[lmin:lmax]))).flatten()
            period[k + ixb] = lmin + tloc - 1
    return vseg, vsl, SF, Ef, period


def calPitch(y, win, step, sr):
    '''
    calculate pitch
    :param y: data of wav file
    :param win: windows
    :param step: inc
    :param sr: frequency of wav file
    :return: pitch Hz, period dot
    '''
    T1 = 0.05
    voicesef, vosl, SF, Ef, period = pitch_Corr(y, win, step, T1, sr)
    # period is pitch period
    pitch = sr / (period + eps)
    pindex = np.where(pitch > 5000)
    pitch[pindex] = 0
    return pitch, period


def JitterAbsolute(pitch):
    period = 1 / (pitch + eps)
    pindex = np.where(period > 5000)
    period[pindex] = 0
    n = len(period)
    sigma = 0
    for i in range(n - 1):
        sigma = np.abs(period[i] - period[i + 1])
    jitter_absolute = sigma / (n - 1)
    return jitter_absolute


def JitterRelative(pitch):
    period = 1 / (pitch + eps)
    pindex = np.where(period > 5000)
    period[pindex] = 0
    n = len(period)
    sigma = 0
    jitter_relative = JitterAbsolute(pitch) / (np.sum(period) / n)


#############################################
##
## calculate formant frequency and bandwidth
##
#############################################
# def Formant_Interpolation(u, sr, p=12):
def stFormant(u, sr, p=12):
    '''
    F: formant frequency
    Bw: formant bandwith
    u: one frame of signal
    p: number of LPC
    sr: sampling rate
	return 
	[1] formant frequency array
	[2] formant bandwidth array
    '''
    ### calculate lpc begin
    a_filter = lpc.autocor(u, p)
    a_filter_num = a_filter.numdict
    i = 0
    a = []
    for k, v in a_filter_num.items():
        if i != k:
            while (i != k):
                a.append(0)
                i = i + 1
        a.append(v)
        i = i + 1
    a = np.array(a)
    ### calculate lpc end
    U = lpcar2pf(a, 255)  # 由LPC系数求出频谱曲线
    df = sr / 512  # 频谱分辨力
    Loc, Mdict = signal.find_peaks(U)  # find peaks in U
    nFormant = len(Loc)
    F = np.zeros(nFormant)  # 共振峰频率
    Bw = np.zeros(nFormant)  # 共振峰带宽
    # 内插法
    i = 0
    for m in Loc:
        m1 = m - 1;
        m2 = m + 1
        p = U[m];
        p1 = U[m1];
        p2 = U[m2]
        aa = (p1 + p2) / 2 - p
        bb = (p2 - p1) / 2
        cc = p
        dm = - bb / (2 * aa)  # 极大值对应频率
        pp = -bb ** 2 / (4 * aa) + cc  # 中心频率对应功率谱
        bf = - np.sqrt(bb ** 2 - 4 * aa * (cc - 0.5 * pp)) / (2 * aa)  # 带宽x轴值
        F[i] = (m + dm) * df
        Bw[i] = 2 * bf * df
        i = i + 1

    return F, Bw


def lpcar2pf(ar, npoints):
    '''
    ar : lpc coefficient
    np : 频谱范围
    return : 频谱曲线
    '''
    return np.abs(np.fft.rfft(ar, 2 * npoints + 2)) ** (-2)


def pre_emphasis(y, coefficient=0.99):
    '''
    y : original signal
    coefficient: emphasis coefficient
    '''
    return np.append(y[0], y[1:] - coefficient * y[:-1])


###############################################################

###################
##
## from opensimle
##
#####################
def stIntensity(frame):
    '''
    cannot understand what differ from energy
    '''
    fn = len(frame)
    hamWin = np.hamming(fn)
    winSum = np.sum(hamWin)
    if winSum <= 0.0:
        winSum = 1.0
    I0 = 0.000001
    Im = 0
    for i in range(fn):
        Im = hamWin[i] * frame[i] ** 2
    intensity = Im/winSum
    loudness = (Im / I0) ** .3
    return intensity, loudness


def main():
    berlin_dataset = Dataset('berlin')
    for wav in os.listdir(berlin_dataset.wav_files):
        wav_file = '%s/%s' % (berlin_dataset.wav_files, wav)
        wav_feature_folder = '%s/%s' % (berlin_dataset.NN_inputs, wav)

        genFeatures(wav_file, wav_feature_folder)


#genFeatures('03a01Fa.wav')
win_size = 0.03
step = 0.01
Fs, x = audioBasicIO.readAudioFile('03a01Fa.wav')
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, win_size * Fs, step * Fs)
