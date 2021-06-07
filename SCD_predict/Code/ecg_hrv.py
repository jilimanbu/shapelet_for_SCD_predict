import numpy as np
import pandas as pd
import sklearn
import nolds
import mne
import biosppy
import scipy.signal

def ecg_hrv(rpeaks=None, rri=None, sampling_rate=1000, hrv_features=["time", "frequency", "nonlinear"]):
    """
    Computes the Heart-Rate Variability (HRV). Shamelessly stolen from the `hrv <https://github.com/rhenanbartels/hrv/blob/develop/hrv>`_ package by Rhenan Bartels. All credits go to him.

    Parameters
    ----------
    rpeaks : list or ndarray
        R-peak location indices.
    rri: list or ndarray
        RR intervals in the signal. If this argument is passed, rpeaks should not be passed.
    sampling_rate : int
        Sampling rate (samples/second).
    hrv_features : list
        What HRV indices to compute. Any or all of 'time', 'frequency' or 'nonlinear'.

    Returns
    ----------
    hrv : dict
        Contains hrv features and percentage of detected artifacts.

    Example
    ----------
    >>> import neurokit as nk
    >>> sampling_rate = 1000
    >>> hrv = nk.bio_ecg.ecg_hrv(rpeaks=rpeaks, sampling_rate=sampling_rate)

    Notes
    ----------
    *Details*

    - **HRV**: Heart-Rate Variability (HRV) is a finely tuned measure of heart-brain communication, as well as a strong predictor of morbidity and death (Zohar et al., 2013). It describes the complex variation of beat-to-beat intervals mainly controlled by the autonomic nervous system (ANS) through the interplay of sympathetic and parasympathetic neural activity at the sinus node. In healthy subjects, the dynamic cardiovascular control system is characterized by its ability to adapt to physiologic perturbations and changing conditions maintaining the cardiovascular homeostasis (Voss, 2015). In general, the HRV is influenced by many several factors like chemical, hormonal and neural modulations, circadian changes, exercise, emotions, posture and preload. There are several procedures to perform HRV analysis, usually classified into three categories: time domain methods, frequency domain methods and non-linear methods.

       - **sdNN**: The standard deviation of the time interval between successive normal heart beats (*i.e.*, the RR intervals). Reflects all influences on HRV including slow influences across the day, circadian variations, the effect of hormonal influences such as cortisol and epinephrine. It should be noted that total variance of HRV increases with the length of the analyzed recording.
       - **meanNN**: The the mean RR interval.
       - **CVSD**: The coefficient of variation of successive differences (van Dellen et al., 1985), the RMSSD divided by meanNN.
       - **cvNN**: The Coefficient of Variation, *i.e.* the ratio of sdNN divided by meanNN.
       - **RMSSD** is the root mean square of the RR intervals (*i.e.*, square root of the mean of the squared differences in time between successive normal heart beats). Reflects high frequency (fast or parasympathetic) influences on HRV (*i.e.*, those influencing larger changes from one beat to the next).
       - **medianNN**: Median of the Absolute values of the successive Differences between the RR intervals.
       - **madNN**: Median Absolute Deviation (MAD) of the RR intervals.
       - **mcvNN**: Median-based Coefficient of Variation, *i.e.* the ratio of madNN divided by medianNN.
       - **pNN50**: The proportion derived by dividing NN50 (The number of interval differences of successive RR intervals greater than 50 ms) by the total number of RR intervals.
       - **pNN20**: The proportion derived by dividing NN20 (The number of interval differences of successive RR intervals greater than 20 ms) by the total number of RR intervals.
       - **Triang**: The HRV triangular index measurement is the integral of the density distribution (that is, the number of all RR intervals) divided by the maximum of the density distribution (class width of 8ms).
       - **Shannon_h**: Shannon Entropy calculated on the basis of the class probabilities pi (i = 1,...,n with n—number of classes) of the NN interval density distribution (class width of 8 ms resulting in a smoothed histogram suitable for HRV analysis).
       - **VLF** is the variance (*i.e.*, power) in HRV in the Very Low Frequency (.003 to .04 Hz). Reflect an intrinsic rhythm produced by the heart which is modulated by primarily by sympathetic activity.
       - **LF**  is the variance (*i.e.*, power) in HRV in the Low Frequency (.04 to .15 Hz). Reflects a mixture of sympathetic and parasympathetic activity, but in long-term recordings like ours, it reflects sympathetic activity and can be reduced by the beta-adrenergic antagonist propanolol (McCraty & Atkinson, 1996).
       - **HF**  is the variance (*i.e.*, power) in HRV in the High Frequency (.15 to .40 Hz). Reflects fast changes in beat-to-beat variability due to parasympathetic (vagal) activity. Sometimes called the respiratory band because it corresponds to HRV changes related to the respiratory cycle and can be increased by slow, deep breathing (about 6 or 7 breaths per minute) (Kawachi et al., 1995) and decreased by anticholinergic drugs or vagal blockade (Hainsworth, 1995).
       - **Total_Power**: Total power of the density spectra.
       - **LFHF**: The LF/HF ratio is sometimes used by some investigators as a quantitative mirror of the sympatho/vagal balance.
       - **LFn**: normalized LF power LFn = LF/(LF+HF).
       - **HFn**: normalized HF power HFn = HF/(LF+HF).
       - **LFp**: ratio between LF and Total_Power.
       - **HFp**: ratio between H and Total_Power.
       - **DFA**: Detrended fluctuation analysis (DFA) introduced by Peng et al. (1995) quantifies the fractal scaling properties of time series. DFA_1 is the short-term fractal scaling exponent calculated over n = 4–16 beats, and DFA_2 is the long-term fractal scaling exponent calculated over n = 16–64 beats.
       - **Shannon**: Shannon Entropy over the RR intervals array.
       - **Sample_Entropy**: Sample Entropy (SampEn) over the RR intervals array with emb_dim=2.
       - **Correlation_Dimension**: Correlation Dimension over the RR intervals array with emb_dim=2.
       - **Entropy_Multiscale**: Multiscale Entropy over the RR intervals array  with emb_dim=2.
       - **Entropy_SVD**: SVD Entropy over the RR intervals array with emb_dim=2.
       - **Entropy_Spectral_VLF**: Spectral Entropy over the RR intervals array in the very low frequency (0.003-0.04).
       - **Entropy_Spectral_LF**: Spectral Entropy over the RR intervals array in the low frequency (0.4-0.15).
       - **Entropy_Spectral_HF**: Spectral Entropy over the RR intervals array in the very high frequency (0.15-0.40).
       - **Fisher_Info**: Fisher information over the RR intervals array with tau=1 and emb_dim=2.
       - **Lyapunov**: Lyapunov Exponent over the RR intervals array with emb_dim=58 and matrix_dim=4.
       - **FD_Petrosian**: Petrosian's Fractal Dimension over the RR intervals.
       - **FD_Higushi**: Higushi's Fractal Dimension over the RR intervals array with k_max=16.

    *Authors*

    - `Dominique Makowski <https://dominiquemakowski.github.io/>`_
    - Rhenan Bartels (https://github.com/rhenanbartels)

    *Dependencies*

    - scipy
    - numpy

    *See Also*

    - RHRV: http://rhrv.r-forge.r-project.org/

    References
    -----------
    - Heart rate variability. (1996). Standards of measurement, physiological interpretation, and clinical use. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. Eur Heart J, 17, 354-381.
    - Voss, A., Schroeder, R., Heitmann, A., Peters, A., & Perz, S. (2015). Short-term heart rate variability—influence of gender and age in healthy subjects. PloS one, 10(3), e0118308.
    - Zohar, A. H., Cloninger, C. R., & McCraty, R. (2013). Personality and heart rate variability: exploring pathways from personality to cardiac coherence and health. Open Journal of Social Sciences, 1(06), 32.
    - Smith, A. L., Owen, H., & Reynolds, K. J. (2013). Heart rate variability indices for very short-term (30 beat) analysis. Part 2: validation. Journal of clinical monitoring and computing, 27(5), 577-585.
    - Lippman, N. E. A. L., Stein, K. M., & Lerman, B. B. (1994). Comparison of methods for removal of ectopy in measurement of heart rate variability. American Journal of Physiology-Heart and Circulatory Physiology, 267(1), H411-H418.
    - Peltola, M. A. (2012). Role of editing of R–R intervals in the analysis of heart rate variability. Frontiers in physiology, 3.
    """
    # Check arguments: exactly one of rpeaks or rri has to be given as input
    if rpeaks is None and rri is None:
        raise ValueError("Either rpeaks or RRIs needs to be given.")

    if rpeaks is not None and rri is not None:
        raise ValueError("Either rpeaks or RRIs should be given but not both.")

    # Initialize empty dict
    hrv = {}

    # Preprocessing
    # ==================
    # Extract RR intervals (RRis)
    if rpeaks is not None:
        # Rpeaks is given, RRis need to be computed
        RRis = np.diff(rpeaks)
    else:
        # Case where RRis are already given:
        RRis = rri


    # Basic resampling to 1Hz to standardize the scale
    RRis = RRis/sampling_rate
    RRis = RRis.astype(float)


    # Artifact detection - Statistical
    for index, rr in enumerate(RRis):
        # Remove RR intervals that differ more than 25% from the previous one
        if RRis[index] < RRis[index-1]*0.75:
            RRis[index] = np.nan
        if RRis[index] > RRis[index-1]*1.25:
            RRis[index] = np.nan

    # Artifact detection - Physiological (http://emedicine.medscape.com/article/2172196-overview)
    RRis = pd.Series(RRis)
    RRis[RRis < 0.6] = np.nan
    RRis[RRis > 1.3] = np.nan

     # Sanity check
    if len(RRis) <= 1:
        print("NeuroKit Warning: ecg_hrv(): Not enough R peaks to compute HRV :/")
        return(hrv)

    # Artifacts treatment
    hrv["n_Artifacts"] = pd.isnull(RRis).sum()/len(RRis)
    artifacts_indices = RRis.index[RRis.isnull()]  # get the artifacts indices
    RRis = RRis.drop(artifacts_indices)  # remove the artifacts


    # Rescale to 1000Hz
    RRis = RRis*1000
    hrv["RR_Intervals"] = RRis  # Values of RRis

    # Sanity check after artifact removal
    if len(RRis) <= 1:
        print("NeuroKit Warning: ecg_hrv(): Not enough normal R peaks to compute HRV :/")
        return(hrv)

    # Time Domain
    # ==================
    if "time" in hrv_features:
        hrv["RMSSD"] = np.sqrt(np.mean(np.diff(RRis) ** 2))
        hrv["meanNN"] = np.mean(RRis)
        hrv["sdNN"] = np.std(RRis, ddof=1)  # make it calculate N-1
        hrv["cvNN"] = hrv["sdNN"] / hrv["meanNN"]
        hrv["CVSD"] = hrv["RMSSD"] / hrv["meanNN"]
        hrv["medianNN"] = np.median(abs(RRis))
        hrv["madNN"] = mad(RRis, constant=1)
        hrv["mcvNN"] = hrv["madNN"] / hrv["medianNN"]
        nn50 = sum(abs(np.diff(RRis)) > 50)
        nn20 = sum(abs(np.diff(RRis)) > 20)
        hrv["pNN50"] = nn50 / len(RRis) * 100
        hrv["pNN20"] = nn20 / len(RRis) * 100






    # Frequency Domain Preparation
    # ==============================
    if "frequency" in hrv_features:

        # Interpolation
        # =================
        # Convert to continuous RR interval (RRi)
        beats_times = rpeaks[1:].copy()  # the time at which each beat occured starting from the 2nd beat
        beats_times -= list(beats_times)[0]  # So it starts at 0
        beats_times = np.delete(list(beats_times), artifacts_indices)  # delete also the artifact beat moments

        try:
            RRi = interpolate(RRis, beats_times, sampling_rate)  # Interpolation using 3rd order spline
        except TypeError:
            print("NeuroKit Warning: ecg_hrv(): Sequence too short to compute interpolation. Will skip many features.")
            return(hrv)


        hrv["df"] = RRi.to_frame("ECG_RR_Interval")  # Continuous (interpolated) signal of RRi



        # Geometrical Method (actually part of time domain)
        # =========================================
        # TODO: This part needs to be checked by an expert. Also, it would be better to have Renyi entropy (a generalization of shannon's), but I don't know how to compute it.
        try:
            bin_number = 32  # Initialize bin_width value
            # find the appropriate number of bins so the class width is approximately 8 ms (Voss, 2015)
            for bin_number_current in range(2, 50):
                bin_width = np.diff(np.histogram(RRi, bins=bin_number_current, density=True)[1])[0]
                if abs(8 - bin_width) < abs(8 - np.diff(np.histogram(RRi, bins=bin_number, density=True)[1])[0]):
                    bin_number = bin_number_current
            hrv["Triang"] = len(RRis)/np.max(np.histogram(RRi, bins=bin_number, density=True)[0])
            hrv["Shannon_h"] = complexity_entropy_shannon(np.histogram(RRi, bins=bin_number, density=True)[0])
        except ValueError:
            hrv["Triang"] = np.nan
            hrv["Shannon_h"] = np.nan



        # Frequency Domain Features
        # ==========================
        freq_bands = {
          "ULF": [0.0001, 0.0033],
          "VLF": [0.0033, 0.04],
          "LF": [0.04, 0.15],
          "HF": [0.15, 0.40],
          "VHF": [0.4, 0.5]}


        # Frequency-Domain Power over time
        freq_powers = {}
        for band in freq_bands:
            freqs = freq_bands[band]
            # Filter to keep only the band of interest
            filtered, sampling_rate, params = biosppy.signals.tools.filter_signal(signal=RRi, ftype='butter', band='bandpass', order=1, frequency=freqs, sampling_rate=sampling_rate)
            # Apply Hilbert transform
            amplitude, phase = biosppy.signals.tools.analytic_signal(filtered)
            # Extract Amplitude of Envelope (power)
            freq_powers["ECG_HRV_" + band] = amplitude

        freq_powers = pd.DataFrame.from_dict(freq_powers)
        freq_powers.index = hrv["df"].index
        hrv["df"] = pd.concat([hrv["df"], freq_powers], axis=1)


        # Compute Power Spectral Density (PSD) using multitaper method
        power, freq = mne.time_frequency.psd_array_multitaper(RRi, sfreq=sampling_rate, fmin=0, fmax=0.5,  adaptive=False, normalization='length')

        def power_in_band(power, freq, band):
            power =  np.trapz(y=power[(freq >= band[0]) & (freq < band[1])], x=freq[(freq >= band[0]) & (freq < band[1])])
            return(power)

        # Extract Power according to frequency bands
        hrv["ULF"] = power_in_band(power, freq, freq_bands["ULF"])
        hrv["VLF"] = power_in_band(power, freq, freq_bands["VLF"])
        hrv["LF"] = power_in_band(power, freq, freq_bands["LF"])
        hrv["HF"] = power_in_band(power, freq, freq_bands["HF"])
        hrv["VHF"] = power_in_band(power, freq, freq_bands["VHF"])
        hrv["Total_Power"] = power_in_band(power, freq, [0, 0.5])

        hrv["LFn"] = hrv["LF"]/(hrv["LF"]+hrv["HF"])
        hrv["HFn"] = hrv["HF"]/(hrv["LF"]+hrv["HF"])
        hrv["LF/HF"] = hrv["LF"]/hrv["HF"]
        hrv["LF/P"] = hrv["LF"]/hrv["Total_Power"]
        hrv["HF/P"] = hrv["HF"]/hrv["Total_Power"]


    # TODO: THIS HAS TO BE CHECKED BY AN EXPERT - Should it be applied on the interpolated on raw RRis?
    # Non-Linear Dynamics
    # ======================
    if "nonlinear" in hrv_features:
        if len(RRis) > 17:
            hrv["DFA_1"] = nolds.dfa(RRis, range(4, 17))
        if len(RRis) > 66:
            hrv["DFA_2"] = nolds.dfa(RRis, range(16, 66))
        hrv["Shannon"] = complexity_entropy_shannon(RRis)
        hrv["Sample_Entropy"] = nolds.sampen(RRis, emb_dim=2)
        try:
            hrv["Correlation_Dimension"] = nolds.corr_dim(RRis, emb_dim=2)
        except AssertionError as error:
            print("NeuroKit Warning: ecg_hrv(): Correlation Dimension. Error: " + str(error))
            hrv["Correlation_Dimension"] = np.nan
        mse = complexity_entropy_multiscale(RRis, max_scale_factor=20, m=2)
        hrv["Entropy_Multiscale_AUC"] = mse["MSE_AUC"]
        hrv["Entropy_SVD"] = complexity_entropy_svd(RRis, emb_dim=2)
        hrv["Entropy_Spectral_VLF"] = complexity_entropy_spectral(RRis, sampling_rate, bands=np.arange(0.0033, 0.04, 0.001))
        hrv["Entropy_Spectral_LF"] = complexity_entropy_spectral(RRis, sampling_rate, bands=np.arange(0.04, 0.15, 0.001))
        hrv["Entropy_Spectral_HF"] = complexity_entropy_spectral(RRis, sampling_rate, bands=np.arange(0.15, 0.40, 0.001))
        hrv["Fisher_Info"] = complexity_fisher_info(RRis, tau=1, emb_dim=2)
#        lyap exp doesn't work for some reasons
#        hrv["Lyapunov"] = np.max(nolds.lyap_e(RRis, emb_dim=58, matrix_dim=4))

        hrv["FD_Petrosian"] = complexity_fd_petrosian(RRis)
        hrv["FD_Higushi"] = complexity_fd_higushi(RRis, k_max=16)

    # TO DO:
    # Include many others (see Voss 2015)


    return(hrv)