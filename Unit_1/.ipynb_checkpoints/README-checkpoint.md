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
![Sampling Rate](/Images/Sampling.png "Sampling Rate")

Note: Ensure that all audio examples in your dataset have the same sampling rate
Transformer models that solve audio tasks treat examples as sequences and rely on attention mechanisms to learn audio or multimodal representation. Since sequences are different for audio examples at different sampling rates, it will be challenging for models to generalize between sampling rates. Resampling is the process of making the sampling rates match.


> **Why is the Nyquist Limit Half the Sampling Rate? What is its Significance?**
To understand the Nyquist limit, consider an everyday example.
Imagine you've drawn a smooth, curvy line (representing an analog signal) on a piece of paper. Now you want to digitize this curve and store it in your computer. To do that, the computer samples points along the curve—taking measurements at fixed intervals.
If you take too few samples, you may miss important details of the curve's shape. This is called aliasing—where higher frequencies in the original signal appear as lower frequencies in the sampled version, distorting the original information.

> **Why Sample at Twice the Highest Frequency?**
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

