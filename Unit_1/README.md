# Introduction
A **sound wave** is a continuous signal that contains infinite signal values in a given time.
However a **digital device**  expects a finite array. Hence sound waves needs to be converted into series of discrete values called **digital representation**.

There are largely 3 common audio file formats that we may encounter -
File Formats - 
- .wav (Waveform Audio File)
- .flac (Free Lossless Audio Codec) 
- .mp3 (MPEG-1 Audio Layer 3)

The general process to convert continuous signal to digital representation is as follows -
- Analog signal is captured by a microphone
- Converts sound wave into electrical signal
- Digitalized by Analog-to-Digital to get digital representation through sampling

## Sampling and Sampling Rate
- Sampling is the process of measuring the value of a continuous signal at fixed time steps. The sampled waveform is discrete, since it contains a finite number of signal values at uniform intervals.
- The sampling rate (also called sampling frequency) is the number of samples taken in one second and is measured in hertz (Hz).
- A common sampling rate used in training speech models is 16,000 Hz or 16 kHz.
- The choice of sampling rate primarily determines the highest frequency that can be captured from the signal. This is also known as the **Nyquist limit** and is exactly **half the sampling rate**. 
- The audible frequencies in human speech are below 8 kHz and therefore sampling speech at 16 kHz is sufficient. Using a higher sampling rate will not capture more information and merely leads to an increase in the computational cost of processing such files. On the other hand, sampling audio at too low a sampling rate will result in information loss. Speech sampled at 8 kHz will sound muffled, as the higher frequencies cannot be captured at this rate.

Note: Ensure that all audio examples in your dataset have the same sampling rate
Transformer models that solve audio tasks treat examples as sequences and rely on attention mechanisms to learn audio or multimodal representation. Since sequences are different for audio examples at different sampling rates, it will be challenging for models to generalize between sampling rates. Resampling is the process of making the sampling rates match.


> **Why is the Nyquist Limit Half the Sampling Rate? What is its Significance?**<br>
To understand the Nyquist limit, consider an everyday example.
Imagine you've drawn a smooth, curvy line (representing an analog signal) on a piece of paper. Now you want to digitize this curve and store it in your computer. To do that, the computer samples points along the curve—taking measurements at fixed intervals.
If you take too few samples, you may miss important details of the curve's shape. This is called aliasing—where higher frequencies in the original signal appear as lower frequencies in the sampled version, distorting the original information.

> **Why Sample at Twice the Highest Frequency?**<br>
According to the Nyquist-Shannon Sampling Theorem, to accurately reconstruct a signal from its samples without aliasing, you must sample at at least twice the highest frequency present in the signal. This threshold is known as the Nyquist rate, and half of the sampling rate is called the Nyquist frequency. Frequencies above this limit cannot be uniquely reconstructed and will be "aliased" into lower frequencies.
Curve Analogy:
Let’s go back to your curve example:
>- Sampling the curve is like taking dots along the line.
>- If the curve has a lot of rapid changes (i.e., high-frequency components), you need more dots to capture those changes accurately.
>- If you place only one dot per cycle of a wave, you might miss where it peaks or dips—leading to a false representation.
>- But if you place two dots per cycle, you can clearly see the wave’s shape—making reconstruction accurate.

>Pizza Analogy:
Imagine a pizza with multiple toppings spread unevenly (pepperoni, olives, mushrooms, etc.). If you look at just one slice, you might see only olives and think that's the only topping. But if you sample multiple slices, you get a clearer picture of all the toppings.
Similarly, in signal sampling:
>- One sample (or slice) isn’t enough to know the "full flavor" of the signal.
>- You need to sample frequently enough to capture all the key details (frequencies).
>- Hence, sampling at less than twice the highest frequency is like judging a pizza by one slice—you’ll get an inaccurate understanding.
>

## Amplitude and Bit Depth
The amplitude of a sound describes the sound pressure level at any given instant and is measured in decibels (dB). 

In digital audio, each audio sample records the amplitude of the audio wave at a point in time. The bit depth of the sample determines with how much precision this amplitude value can be described. The higher the bit depth, the more faithfully the digital representation approximates the original continuous sound wave.

Floating-point audio samples are expected to lie within the [-1.0, 1.0] range. Since machine learning models naturally work on floating-point data, the audio must first be converted into floating-point format before it can be used to train the model.

The decibel scale for real-world audio starts at 0 dB, which represents the quietest possible sound humans can hear, and louder sounds have larger values. However, for digital audio signals, 0 dB is the loudest possible amplitude, while all other amplitudes are negative. As a quick rule of thumb: every -6 dB is a halving of the amplitude, and anything below -60 dB is generally inaudible unless you really crank up the volume.

> Machines take max value as 0 db (glass completely full). Anything over it is like spilling of water from the full glass. You can only reduce the amount of water. Hence -inf db is silence in machines while 0 db is the loudest.
> We need to set max limit to machines and hence taking 0 dB the loudest possible emplitude as compared to inf is easier.

## Audio as Waveform
Plots the sample values over time and illustrates the changes in the sound’s amplitude. This is also known as the time domain representation of sound.
This type of visualization is useful for identifying specific features of the audio signal such as the timing of individual sound events, the overall loudness of the signal, and any irregularities or noise present in the audio.

## The Frequency Spectrum
The spectrum is computed using the discrete Fourier transform or DFT. It describes the individual frequencies that make up the signal and how strong they are.

This plots the strength of the various frequency components that are present in this audio segment. The frequency values are on the x-axis, usually plotted on a logarithmic scale, while their amplitudes are on the y-axis.

The frequency spectrum that we plotted in Introduction.ipynb shows several peaks. These peaks correspond to the harmonics of the note that’s being played, with the higher harmonics being quieter.

*Where the waveform plots the amplitude of the audio signal over time, the spectrum visualizes the amplitudes of the individual frequencies at a fixed point in time.*

## Spectrogram
Spectrum only shows a frozen snapshot of the frequencies at a given instant.
Take multiple DFTs, each covering only a small slice of time, and stack the resulting spectra together. This gives us the spectrogram.
A spectrogram plots the frequency content of an audio signal as it changes over time. It allows you to see time, frequency, and amplitude all on one graph. The algorithm that performs this computation is the STFT or Short Time Fourier Transform.

The x-axis represents time as in the waveform visualization but now the y-axis represents frequency in Hz. The intensity of the color gives the amplitude or power of the frequency component at each point in time, measured in decibels (dB).

The spectrogram is created by taking short segments of the audio signal, typically lasting a few milliseconds, and calculating the discrete Fourier transform of each segment to obtain its frequency spectrum. The resulting spectra are then stacked together on the time axis to create the spectrogram.

> Since the spectrogram and the waveform are different views of the same data, it’s possible to turn the spectrogram back into the original waveform using the inverse STFT. However, this requires the phase information in addition to the amplitude information. If the spectrogram was generated by a machine learning model, it typically only outputs the amplitudes. In that case, we can use a phase reconstruction algorithm such as the classic Griffin-Lim algorithm, or using a neural network called a vocoder, to reconstruct a waveform from the spectrogram.

## Mel Spectrogram
Variant of spectrogram commonly used in speech. 

In a standard spectrogram, the frequency axis is linear and is measured in hertz (Hz). However, the human auditory system is more sensitive to changes in lower frequencies than higher frequencies, and this sensitivity decreases logarithmically as frequency increases. The mel scale is a perceptual scale that approximates the non-linear frequency response of the human ear.
To create a mel spectrogram, the STFT is used just like before, splitting the audio into short segments to obtain a sequence of frequency spectra. Additionally, each spectrum is sent through a set of filters, the so-called mel filterbank, to transform the frequencies to the mel scale.

Just as with a regular spectrogram, it’s common practice to express the strength of the mel frequency components in decibels. This is commonly referred to as a log-mel spectrogram, because the conversion to decibels involves a logarithmic operation. 

> Note: There are two diff mel scale: 'htk' and 'slaney'
Moreover instead of power spectrogram, amplitude spectrogram may be used. 
The conversion to a log-mel spectrogram doesn't always compute true decibels but may simply take the `log`. Therefore, if a machine learning model expects a mel spectrogram as input, double check to make sure you're computing it the same way.

