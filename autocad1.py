# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 19:07:29 2021

@author: jiach
"""

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_roc_curve, auc, accuracy_score, f1_score
from scipy.signal import butter, lfilter,iirnotch,resample_poly
import pywt
import neurokit2 as nk
import pyhrv
import csv
import streamlit as st
import joblib
import pandas as pd
from io import BytesIO
from PIL import Image

##########################################################################
################   This is the code of the program   #####################
##########################################################################

#define butterworth filtering function

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#define band-reject filtering function
def band_notch(freq, quality, fs):
    nyq = 0.5 * fs
    w0=freq/nyq
    b, a = iirnotch(w0, quality)
    return b, a


def notch_filter(data, freq,quality,fs):
    b, a = band_notch(freq, quality, fs)
    y = lfilter(b, a, data)
    return y

#define upsampling function for CAD patients
def upsamplingCAD(data,freq):
    datalength = len(data)
    num = int(datalength*freq/250)
    y = resample_poly(data,freq,250)
    return y
st.set_page_config(layout="wide")
st.title('Automated Coronary Artery Disease (CAD) Diagnosis Using Electrocardiogram (ECG)')
st.subheader('Made by Jiachen Guo and Ashwin Vazhayil')

st.title('What are CAD and ECG?')
st.header('')
image = Image.open('background.jpg')
st.image(image,width=1300)

st.image("https://drive.google.com/uc?export=download&id=1t0kCYR0ueILlz2JaNPddFHxqGdR1fxUf")
st.markdown('[1] www.alilamedicalmedia.com')
st.header('')
image = Image.open('outline.jpg')
st.image(image,width=1100)
st.markdown('Download ECG sample file 1 for healthy people: https://drive.google.com/uc?export=download&id=1FrCRrY3xPd2rxknkPMdQNw1DN4_lPdyJ')
st.markdown('Download ECG sample file 2 for healthy people: https://drive.google.com/uc?export=download&id=1cG95GGSrZ1EjoxLmehWyJfhkhOTfWbNV')
st.markdown('Download ECG sample file 1 for CAD patient: https://drive.google.com/uc?export=download&id=1vDrn7pyexfAaC4Pmu3P6ZbcI3Hed1akI')
st.markdown('Download ECG sample file 2 for CAD patient: https://drive.google.com/uc?export=download&id=1dYvOE-TUp7pnuMABis4u-8_nr8KC_h-L')
col1,col2,col3=st.columns([1,0.1,1])
#parameters used in filtering
lowcut = 0.3
highcut = 15
reject=50 #powerline inteference noise  
quality=20 #doesn't require change
with col1:
    st.title('Healthy people case')
    #%% ecg data import
    uploaded_file = st.file_uploader('Please select a csv file which contains ECG data to upload')
    if uploaded_file is not None:
        reader = pd.read_csv(uploaded_file,names=['Time (s)','Voltage (mv)'])
    #    data = list(reader)
        data=np.asarray(reader)
        st.markdown('ECG data has been loaded successfully...')
        st.write(reader)
        t=data[:,0]
        s=data[:,1]
        s0=s[1000:-1]
        fs=1/(t[1]-t[0])
        s=butter_bandpass_filter(s, lowcut, highcut, fs, order=1)
        t=t[1000:-1]-t[1000]
        s=s[1000:-1]
        
        #%% show denoised ECG (data collection)
    
        ecg=[] #ecg voltage matrix
        ecgt=[] #ecg time matrix
        rp=[] #index matrix for all r peaks
        rpt=[] #r peak corresponding time
        dur=[] #R peak duration matrix 
        #################
        #######################


        
        st.title('ECG visulization (healthy people)')
        pt = st.slider('Scroll to show the imported ECG', 0.0, t[-1]-5, 0.01)
        figecg, ax1 = plt.subplots(figsize=(9,4))
        ax1.plot(t,s)
        plt.xlim(pt,pt+5)
        plt.ylim(-2,4)
        plt.title('ECG data for healthy people', fontsize=16)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Voltage (mV)', fontsize=14)
        st.pyplot(figecg)

        
        
        #%% R peaks detection
        st.title('R-peak detection (healthy people)')
        
        signals, info = nk.ecg_peaks(s,sampling_rate=250)
        
        pt = st.slider('Scroll to show detected R peaks', 0.0, t[-1]-5, 0.01)
        
        figecg, ax1 = plt.subplots(figsize=(9,4))
        ax1.plot(t,s)
        r_peaks=info['ECG_R_Peaks']
        
        
        rp.append(r_peaks)
        rptime = t[r_peaks]
        rpt.append(rptime)
        #compute duration matrix
        rptime1=np.delete(rptime,-1)    
        rptime2=np.delete(rptime,1)
        dutime=rptime2-rptime1
        dur.append(dutime)
        
        
        
        
        plt.plot(t[r_peaks],s[r_peaks],'ro')
        plt.xlim(pt,pt+5)
        plt.ylim(-2,4)

        
        
        plt.title('R peak detection for healthy people', fontsize=16)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Voltage (mV)', fontsize=14)
        st.pyplot(figecg)        
        st.subheader('Using detected R peaks, heartbeat durations can be calculated')
        
        
        #%% heart rate variability
        
        st.title('Heart rate variability (HRV) for healthy people')
        
        
        st.subheader('HRV can be calculated using the equation below:')
        st.latex(r''' 
                 HRV_{i}=\frac{60}{\Delta_{t_{R_i}}}
                            ''')
        a1=dur[0]
        a1=a1[1:-1] # duration of each different heart beat the 1st element is removed since it's 0
        hrv1=60/a1 #calculate heart rate variability
                    
        pt = st.slider('Scroll to show derived HRV', 0, len(hrv1)-100, 0)       
        fighrv, axhrv = plt.subplots(figsize=(9,4))
        plt.plot(np.arange(0,len(hrv1)),hrv1)
        plt.title('HRV for healthy people', fontsize=16)
        plt.xlabel('Sample',fontsize=14)
        plt.ylabel('Heart rate (bpm)', fontsize=14)
        plt.xlim(pt,pt+100)
        plt.ylim(40,170)
        st.pyplot(fighrv)
        #%% time domain features
        st.title('Features from time domain (healthy people)')
        st.subheader('Mean heartbeat duration:') 
        st.latex(r'''
                            \bar{\Delta}_{t_R}=\frac{1}{n}\sum_{i=1}^{n} \Delta_{t_{R_i}} 
                            ''')
        st.subheader('Standard deviation of heartbeat duration (SD):')
        st.latex(r'''
                            S D=\sqrt{\frac{1}{n-1} \sum_{i=1}^{n}\left(\Delta_{t_{R_i}}-\overline{\Delta}_{t_R}\right)^{2}}  
                            ''')
        st.subheader('Standard deviation of successive differences (SDSD):')
        st.latex(r'''
                             S D S D=\sqrt{\frac{1}{n-1} \sum_{i=1}^{n}\left(\Delta\left(\Delta_{t_{R_i}}\right)-\overline{\Delta\left(\Delta_{t_R}\right)}\right)^{2}}  
                            ''')          
        
        
        #%% time-domain features
        mean=np.mean(a1) #mean of RR interval duration
        sdnn=pyhrv.time_domain.sdnn(nni=a1*1000)  #nni is in ms
        sdnn=sdnn['sdnn']/1000 #standard deviation of RR interval duration
        sdsd=pyhrv.time_domain.sdsd(nni=a1*1000)  #nni is in ms
        sdsd=sdsd['sdsd']/1000 #standard deviation of RR interval duration differences
        timeHealth=[mean,sdnn,sdsd] # time domain features storation
        
        if st.button('Show extracted time-domain features for healthy people'):
            
            st.markdown('Features extracted from the time domain are summarized below:')
            featuret=np.reshape(timeHealth,(1,-1))
            ftframe=pd.DataFrame(featuret,columns=['Mean','SD','SDSD'])
            st.write(ftframe)
    
    #%% freq-domain features
        st.title('Features from frequency domain (healthy people)')
        st.subheader('''Power distribution in the freq domain reveals how heart rate is controlled by the nervous system''')
        st.subheader('Total power: detect abnormal autonomic activity')
        st.subheader('Low-freq (LF) (0.04~0.15Hz) power: detect sympathetic modulation')
        st.subheader('High-freq (HF) (0.15~0.4Hz) power: detect parasympathetic modulation')
        st.subheader('Ratio LF/HF: reflect sympathetic/parasympathetic balance')
        freq_all=pyhrv.frequency_domain.welch_psd(nni=a1*1000,show_param=False,figsize=(9,4))
        ptotal=freq_all['fft_total']/1000**2 #total power
        pLF=freq_all['fft_norm'][0] #normalized power for low frequency band
        pHF=freq_all['fft_norm'][1] #normalized power for high frequency band https://pyhrv.readthedocs.io/en/latest/_pages/api/frequency.html#welch-s-method-welch-psd
        ratio=freq_all['fft_ratio'] #LF/HF ratio
        freqplot=freq_all['fft_plot']
        st.pyplot(freqplot)
        freqHealth=[ptotal,pLF,pHF,ratio] # freq domain features storation
        if st.button('Show extracted frequency-domain features for healthy people'):
            tF=freq_all['fft_total']
            lowF=tF*freq_all['fft_rel'][1]/100
            highF=tF*freq_all['fft_rel'][2]/100
            featuref=[tF,lowF,highF,ratio]
            featuref=np.reshape(featuref,(1,-1))
            ffframe=pd.DataFrame(featuref,columns=['Total power','Low-frequency power','High-frequency power','LF/HF'])
            st.markdown('Features extracted from the frequency domain are summarized below:')
            st.write(ffframe)   
    
    #%% time-freq features
        st.title('Features from time-frequency domain (healthy people)')
        st.subheader('Discrete wavelet transform of HRV into 4 coefficients: D1~D3, A3')
        st.subheader('Shannon entropy: measure data uncertainty and variability')
        st.subheader('Approximation entropy: quantify data regularity and unpredictability')
        st.subheader('Sampling entropy: assess complexities of physiological signals')
        ca3,cd3,cd2,cd1=pywt.wavedec(hrv1, 'haar', level=3)
        ya3=pywt.waverec([ca3,np.zeros_like(cd3),np.zeros_like(cd2),np.zeros_like(cd1)], 'haar')
        yd3=pywt.waverec([np.zeros_like(ca3),cd3,np.zeros_like(cd2),np.zeros_like(cd1)], 'haar')
        yd2=pywt.waverec([np.zeros_like(ca3),np.zeros_like(cd3),cd2,np.zeros_like(cd1)], 'haar')
        yd1=pywt.waverec([np.zeros_like(ca3),np.zeros_like(cd3),np.zeros_like(cd2),cd1], 'haar')
                                    
        figwave, axwave = plt.subplots(nrows=4, ncols=1, figsize=(9,4))
        plt.subplot(5,1,1)
        plt.plot(hrv1)
        plt.ylim(40,170)
        plt.ylabel('HR/bpm',fontsize=12)
        plt.title('Wavelet decomposition for healthy people',fontsize=16)
        plt.subplot(5,1,2)
        plt.plot(ya3)
        plt.ylim(40,150)
        plt.ylabel('A3',fontsize=12)
        plt.subplot(5,1,3)
        plt.plot(yd3)
        plt.ylim(-50,50)
        plt.ylabel('D3',fontsize=12)
        plt.subplot(5,1,4)
        plt.plot(yd2)
        plt.ylim(-50,50)
        plt.ylabel('D2',fontsize=12)
        plt.subplot(5,1,5)
        plt.plot(yd1)
        plt.ylim(-50,50)
        plt.ylabel('D1',fontsize=12)
        plt.xlabel('Sample',fontsize=12)
        st.pyplot(figwave)
                                    
        waveHealth=[nk.entropy_shannon(ca3),nk.entropy_shannon(cd3)
                                    ,nk.entropy_shannon(cd2)
                                    ,nk.entropy_shannon(cd1)
                                    ,nk.entropy_approximate(ca3)
                                    ,nk.entropy_approximate(cd3)
                                    ,nk.entropy_approximate(cd2)
                                    ,nk.entropy_approximate(cd1)
                                    ,nk.entropy_sample(ca3)
                                    ,nk.entropy_sample(cd3)
                                    ,nk.entropy_sample(cd2)
                                    ,nk.entropy_sample(cd1)]
    
        timeHealth=np.reshape(timeHealth,(1, -1))
        freqHealth=np.reshape(freqHealth,(1, -1))
        waveHealth=np.reshape(waveHealth,(1, -1))
        feature=np.hstack((timeHealth,freqHealth,waveHealth))
        feature=np.reshape(feature,(1, -1))
        fwframe=pd.DataFrame(waveHealth,columns=['Shannon entropy_CD3','Shannon entropy_CD2','Shannon entropy_CD1','Shannon entropy_CA3'
                                                                    ,'Approximate entropy_CD3','Approximate entropy_CD2','Approximate entropy_CD1','Approximate entropy_CA3'
                                                                    ,'Sampling entropy_CD3','Sampling entropy_CD2','Sampling entropy_CD1','Sampling entropy_CA3'])
        
        if st.button('Show extracted time-frequency-domain features for healthy people'):
            st.markdown('Features extracted from the time-frequency domain are summarized below:')
            st.write(fwframe)
        
        #%%Classification
        st.title('Classification using support vector machine (SVM) for healthy people') 
        
        scaler = joblib.load('data_scaler.pkl') 
        X = scaler.transform(feature)
        clf = joblib.load('trainedCADmodel.pkl')
        result=clf.predict(X)
        st.subheader('With features extracted from different domains, pre-trained SVM model can be used for CAD diagnosis')
        if st.button('Show diagnosis result'):
            st.markdown("""
                                        <style>
                                        .noCAD {
                                            font-size:25px !important;color:white;font-weight: 700;background-color: lightgreen;border-radius: 0.4rem;
                                        color: white;
                                        padding: 0.5rem;
                                        margin-bottom: 1rem;
                                        }
                                        </style>
                                        """, unsafe_allow_html=True)
                                        
            st.markdown("""
                                        <style>
                                        .re {
                                            font-size:20px !important;font-weight: 700
                                        }
                                        </style>
                                        """, unsafe_allow_html=True)
                                        
            st.markdown("""
                                        <style>
                                        .CAD {
                                            font-size:30px !important;color:white;font-weight: 700;background-color: lightcoral
                                        }
                                        </style>
                                        """, unsafe_allow_html=True)
            st.markdown('<p class="re">Diagnosis result: </p>',unsafe_allow_html=True)
            if result==1:
                st.markdown('<p class="noCAD">Coronary Artery Disease is not detected! </p>',unsafe_allow_html=True)
            else:
                st.markdown('<p class="CAD">Coronary Artery Disease is detected </p> ',unsafe_allow_html=True)
            
            
        
            

        
        
    else:
        st.markdown('Please choose a valid csv file which contains ECG data')
        st.stop()
    
        #%%
    




with col3:
    st.title('CAD patient case')
    uploaded_file = st.file_uploader('Please select a csv file which contains ECG data to upload1')
    if uploaded_file is None:
        st.markdown('Please choose a valid csv file which contains ECG data')
        st.stop()
    else:
        reader = pd.read_csv(uploaded_file,names=['Time (s)','Voltage (mv)'])
    #    data = list(reader)
        data=np.asarray(reader)
        st.markdown('ECG data has been loaded successfully...')
        st.write(reader)
    
        t=data[:,0]
        s=data[:,1]
        s0=s[1000:-1]
        fs=1/(t[1]-t[0])
        s=butter_bandpass_filter(s, lowcut, highcut, fs, order=1)
        t=t[1000:-1]-t[1000]
        s=s[1000:-1]

        #%% show denoised ECG (data collection)
    
        ecg=[] #ecg voltage matrix
        ecgt=[] #ecg time matrix
        rp=[] #index matrix for all r peaks
        rpt=[] #r peak corresponding time
        dur=[] #R peak duration matrix 
        #################
        #######################



        
        st.title("ECG visulization (CAD patients)")
        pt = st.slider('Scroll to show the imported ECG', 0.0, t[-1]-5, 0.01)
        figecg, ax1 = plt.subplots(figsize=(9,4))
        ax1.plot(t,s)
        plt.xlim(pt,pt+5)
        plt.ylim(-2,4)
        plt.title('ECG data for CAD patients', fontsize=16)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Voltage (mV)', fontsize=14)
        st.pyplot(figecg)
        
        #%% R peaks detection
        st.title("R-peak detection (CAD patients)")
        
        signals, info = nk.ecg_peaks(s,sampling_rate=250)
        
        pt = st.slider('Scroll to show detected R peaks', 0.0, t[-1]-5, 0.01)
        
        figecg, ax1 = plt.subplots(figsize=(9,4))
        ax1.plot(t,s)
        plt.ylim(-2,4)
        r_peaks=info['ECG_R_Peaks']
        
        rp.append(r_peaks)
        rptime = t[r_peaks]
        rpt.append(rptime)
        #compute duration matrix
        rptime1=np.delete(rptime,-1)    
        rptime2=np.delete(rptime,1)
        dutime=rptime2-rptime1
        dur.append(dutime)
        
        
        plt.plot(t[r_peaks],s[r_peaks],'ro')
        plt.xlim(pt,pt+5)

        
        
        plt.title('R peak detection for CAD patients', fontsize=16)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Voltage (mV)', fontsize=14)

        st.pyplot(figecg)
        st.header('')
        st.markdown('')
        #%% heart rate variability
        st.title('Heart rate variability for CAD patients')

        st.header('')
        a2=dur[0]
        a2=a2[1:-1] # duration of each different heart beat the 1st element is removed since it's 0
        hrv2=60/a2 #calculate heart rate variability
        
        st.title('')  
        st.title('')      
        st.title('') 
        pt = st.slider('Scroll to show derived HRV', 0, len(hrv2)-100, 0)       
        fighrv, axhrv = plt.subplots(figsize=(9,4))
        plt.plot(np.arange(0,len(hrv2)),hrv2)
        plt.title('HRV for CAD patients', fontsize=16)
        plt.xlabel('Sample',fontsize=14)
        plt.ylabel('Heart rate (bpm)', fontsize=14)
        plt.xlim(pt,pt+100)
        plt.ylim(40,170)
        st.pyplot(fighrv)    
        
        

        


        #%% time-domain features
        st.title('Features from time domain (CAD patients)')
        st.title('')
        st.title('')
        st.title('')
        st.title('')
        st.title('')
        st.title('')
        st.title('')
        st.title('')
        st.title('')
        st.header('')
        st.text('')

        mean=np.mean(a2) #mean of RR interval duration
        sdnn=pyhrv.time_domain.sdnn(nni=a2*1000)  #nni is in ms
        sdnn=sdnn['sdnn']/1000 #standard deviation of RR interval duration
        sdsd=pyhrv.time_domain.sdsd(nni=a2*1000)  #nni is in ms
        sdsd=sdsd['sdsd']/1000 #standard deviation of RR interval duration differences
        timeHealth=[mean,sdnn,sdsd] # time domain features storation
        
        if st.button('Show extracted time-domain features for CAD patients'):
            
    
            featuret=np.reshape(timeHealth,(1,-1))
            ftframe=pd.DataFrame(featuret,columns=['Mean','SD','SDSD'])
            st.write(ftframe)
        st.text('')
    #%% freq-domain features
        st.title('Features from frequency domain (CAD patients)')
        st.title('')
        st.header('')
        st.header('')
        st.header('')
        st.header('')
        st.header('')
        st.subheader('')
   
        freq_all=pyhrv.frequency_domain.welch_psd(nni=a2*1000,show_param=False,figsize=(9,4))
        ptotal=freq_all['fft_total']/1000**2 #total power
        pLF=freq_all['fft_norm'][0] #normalized power for low frequency band
        pHF=freq_all['fft_norm'][1] #normalized power for high frequency band https://pyhrv.readthedocs.io/en/latest/_pages/api/frequency.html#welch-s-method-welch-psd
        ratio=freq_all['fft_ratio'] #LF/HF ratio
        freqplot=freq_all['fft_plot']
        st.pyplot(freqplot)
        freqHealth=[ptotal,pLF,pHF,ratio] # freq domain features storation
        if st.button('Show extracted frequency-domain features for CAD patients'):
            tF=freq_all['fft_total']
            lowF=tF*freq_all['fft_rel'][1]/100
            highF=tF*freq_all['fft_rel'][2]/100
            featuref=[tF,lowF,highF,ratio]
            featuref=np.reshape(featuref,(1,-1))
            ffframe=pd.DataFrame(featuref,columns=['Total power','Low-frequency power','High-frequency power','LF/HF'])
            st.write(ffframe)  
        st.markdown('')
    #%% time-freq features
        st.title('Features from time-frequency domain (CAD patients)')
        st.header('')
        st.header('')
        st.header('')
        st.header('')
        st.subheader('')
        st.markdown('')
        ca3,cd3,cd2,cd1=pywt.wavedec(hrv2, 'haar', level=3)
        ya3=pywt.waverec([ca3,np.zeros_like(cd3),np.zeros_like(cd2),np.zeros_like(cd1)], 'haar')
        yd3=pywt.waverec([np.zeros_like(ca3),cd3,np.zeros_like(cd2),np.zeros_like(cd1)], 'haar')
        yd2=pywt.waverec([np.zeros_like(ca3),np.zeros_like(cd3),cd2,np.zeros_like(cd1)], 'haar')
        yd1=pywt.waverec([np.zeros_like(ca3),np.zeros_like(cd3),np.zeros_like(cd2),cd1], 'haar')
                                    
        figwave, axwave = plt.subplots(nrows=4, ncols=1, figsize=(9,4))
        plt.subplot(5,1,1)
        plt.plot(hrv2)
        plt.ylim(40,170)
        plt.ylabel('HR/bpm',fontsize=12)
        plt.title('Wavelet decomposition for CAD patients',fontsize=16)
        plt.subplot(5,1,2)
        plt.plot(ya3)
        plt.ylim(40,150)
        plt.ylabel('A3',fontsize=12)
        plt.subplot(5,1,3)
        plt.plot(yd3)
        plt.ylim(-50,50)
        plt.ylabel('D3',fontsize=12)
        plt.subplot(5,1,4)
        plt.plot(yd2)
        plt.ylim(-50,50)
        plt.ylabel('D2',fontsize=12)
        plt.subplot(5,1,5)
        plt.plot(yd1)
        plt.ylim(-50,50)
        plt.ylabel('D1',fontsize=12)
        plt.xlabel('Sample',fontsize=12)
        st.pyplot(figwave)
                                    
        waveHealth=[nk.entropy_shannon(ca3),nk.entropy_shannon(cd3)
                                    ,nk.entropy_shannon(cd2)
                                    ,nk.entropy_shannon(cd1)
                                    ,nk.entropy_approximate(ca3)
                                    ,nk.entropy_approximate(cd3)
                                    ,nk.entropy_approximate(cd2)
                                    ,nk.entropy_approximate(cd1)
                                    ,nk.entropy_sample(ca3)
                                    ,nk.entropy_sample(cd3)
                                    ,nk.entropy_sample(cd2)
                                    ,nk.entropy_sample(cd1)]
    
        feature=np.hstack((timeHealth,freqHealth,waveHealth))
        feature=np.reshape(feature,(1, -1))
        featurew=np.hstack(waveHealth)
        featurew=np.reshape(featurew,(1, -1))
        fwframe=pd.DataFrame(featurew,columns=['Shannon entropy_CD3','Shannon entropy_CD2','Shannon entropy_CD1','Shannon entropy_CA3'
                                                                    ,'Approximate entropy_CD3','Approximate entropy_CD2','Approximate entropy_CD1','Approximate entropy_CA3'
                                                                    ,'Sampling entropy_CD3','Sampling entropy_CD2','Sampling entropy_CD1','Sampling entropy_CA3'])
        
        if st.button('Show extracted time-frequency-domain features for CAD patient'):
            st.markdown('Features extracted from the time-frequency domain are summarized below:')
            st.write(fwframe)
        #%%Classification
        st.title('Classification using support vector machine (SVM) for CAD patients') 
        
        scaler = joblib.load('data_scaler.pkl') 
        X = scaler.transform(feature)
        clf = joblib.load('trainedCADmodel.pkl')
        result=clf.predict(X)
        st.subheader('With features extracted from different domains, pre-trained SVM model can be used for CAD diagnosis')
        if st.button('Show diagnosis result..'):
            st.markdown("""
                                        <style>
                                        .noCAD {
                                            font-size:25px !important;color:white;font-weight: 700;background-color: lightgreen;border-radius: 0.4rem;
                                        color: white;
                                        padding: 0.5rem;
                                        margin-bottom: 1rem;
                                        }
                                        </style>
                                        """, unsafe_allow_html=True)
                                        
            st.markdown("""
                                        <style>
                                        .re {
                                            font-size:20px !important;font-weight: 700
                                        }
                                        </style>
                                        """, unsafe_allow_html=True)
                                        
            st.markdown("""
                                        <style>
                                        .CAD {
                                            font-size:30px !important;color:white;font-weight: 700;background-color: lightcoral
                                        }
                                        </style>
                                        """, unsafe_allow_html=True)
            st.markdown('<p class="re">Diagnosis result: </p>',unsafe_allow_html=True)
            if result==1:
                st.markdown('<p class="noCAD">Coronary Artery Disease is not detected! </p>',unsafe_allow_html=True)
            else:
                st.markdown('<p class="CAD">Coronary Artery Disease is detected </p> ',unsafe_allow_html=True)
            
            
        
            

        
        

    
        
        
