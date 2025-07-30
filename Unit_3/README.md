# Transformer Architecture for Audio
Transformer models used for Audio is pretty similar to the tranformers used for text translation. The only difference is in the input and output -
* Automatic speech recognition (ASR): The input is speech, the output is text.
* Speech synthesis (TTS): The input is text, the output is speech.
* Audio classification: The input is audio, the output is a class probability — one for each element in the sequence or a single class probability for the entire sequence.
* Voice conversion or speech enhancement: Both the input and output are audio.

In case of audio input/output we need to consider the form in which an audio is used - that is its raw form in waveform or in a more processed form of spectrogram.

### Model Input
**Text Input**<br>
It is processed in a similar manner to any other NLP model. 
Steps -
* Text is converted into a sequence of tokens
* Tokens are sent through input embedding layer to convert them into 512-dimensional vectors
* This is then passed into transformer encoder

**Waveform Input**<br>
Models like Wave2Vec and HuBERT use raw audio waveform.
A waveform is a one-dimensional sequence of floating-point numbers, where each number represents the sampled amplitude at a given time.
Steps -
* Normalized to zero mean and unit variance (This helps standardize audio samples across different volumes)
* A small convolutional neural network, known as the feature encoder is then used to turn the sequence of audio samples is turned into an embedding. It reduces the sequence length, until the final convolutional layer outputs a 512-dimensional vector with the embedding for each 25 ms of audio.
* Sent to transformer.

> Drawback: They have long sequence lengths. For example, thirty seconds of audio at a sampling rate of 16 kHz gives an input of length 30 * 16000 = 480000. Longer sequence lengths require more computations in the transformer model, and so higher memory usage.

**Spectrogram Input**<br>
Models such as Whisper first convert the waveform into log-mel spectrogram.
Steps -
* Splits audio into 30s segments and the log-mel spectrogram for each segment has shape (80, 3000) where 80 is the number of mel bins and 3000 is the sequence length. 
* The log-mel spectrogram is then processed by a small CNN into a sequence of embeddings, which goes into the transformer as usual.

By converting to a log-mel spectrogram we’ve reduced the amount of input data, but more importantly, this is a much shorter sequence than the raw waveform. 

### Model Output
**Text Output** <br>
Done my adding a language modeling head — typically a single linear layer — followed by a softmax on top of the transformer’s output. This predicts the probabilities over the text tokens in the vocabulary.

**Spectrogram Output**<br>
To generate audio, it is common to generate a spectrogram and then use additional such as vocoder to turn this spectrogram to waveform. 

Example: In the SpeechT5 TTS model, the output from the transformer network is a sequence of 768-element vectors. A linear layer projects that sequence into a log-mel spectrogram. A post-net, made up of additional linear and convolutional layers, refines the spectrogram by reducing noise. The vocoder then makes the final audio waveform.

**Waveform** <br>
We currently don't have models for this

> If you take an existing waveform and apply the Short-Time Fourier Transform or STFT, it is possible to perform the inverse operation, the ISTFT, to get the original waveform again. This works because the spectrogram created by the STFT contains both amplitude and phase information, and both are needed to reconstruct the waveform. However, audio models that generate their output as a spectrogram typically only predict the amplitude information, not the phase. To turn such a spectrogram into a waveform, we have to somehow estimate the phase information. That's what a vocoder does.

## Connectionist Temporal Classification (CTC) Architectures
* Used with encoder only transformer models for ASR
* Eg. Wav2Vec2, HuBERT and M-CTC-T.
* Encoder only transformer reads the input sequence (audio waveform) and maps this into a sequence of hidden states known as output embeddings. 
* On applying additional linear mapping on sequence of hidden states we can get the class label predictions. 
* The class labels are the characters of the alphabet (a, b, c, …), which allows us to predict any word in the target language with small classification head.

> **Alignment**: We know that the order the speech is spoken in is the same as the order that the text is transcribed in (the alignment is so-called monotonic), but we don’t know how the characters in the transcription line up to the audio. This is where the CTC algorithm comes in.

* In CTC in additional to letters (uppercase or lowecase), we add a padding token (which allows to combine multiple examples in a batch and predicts silence) and a seperator token (to mark the boundaries). 

### CTC in Wave2Vec2
* Raw audio waveform is a 1D time-series signal.
* The CNN feature encoder processes it to downsample and extract high-level features.
    * For example, 1 second of audio becomes a sequence of 50 vectors.
    * That means 1 vector every 20 ms.
    * Each vector actually represents ~25 ms of audio due to overlapping windows.
* The CNN outputs a sequence of hidden states, e.g., shape (50, 768).
* This goes through a Transformer encoder, which models long-range dependencies and context.
* Output remains (50, 768) — 50 time steps, each with a 768-dimensional vector.
* A linear layer (called the CTC head) maps the transformer’s 768-dimensional output to character logits.
    * Eg. for a vocab of size 32, the logit shape becomes (50, 32)
    * So for every 20 ms (50 per second), the model outputs a probability distribution over 32 characters.
* You might see repeated characters because the model predicts something for every 20 ms, even if it’s the same phoneme stretched out.
* To rectify this problem CTC is used. It  aligns model outputs to the target sequence without requiring exact timing.
* It uses **blank token `(_)`** to serve as a hard boundary between groups of characters. The `|` token is used as word seperator.
    * Eg. for an output like `"HE_LLL_LL_OOO||W_OOO_RRR_LD"`
    * CTC will filter out the duplicates using blank token → `"HE_L_L_O||W_O_R_L_D"`
    * Remove blank tokens → `"HELLO WORLD"`
* **Drawback:** It may output words that sound correctly but not spelled correctly. To improve this, use external language model as spell checker.

|Parameter | Wave2Vec2 | HuBERT | M-CTC-T |
|----------|-----------|--------|---------|
|Input | Raw Waveform | Raw Waveform | Mel Spectrogram |
|Training Objective | Predicts speech units for masked part of speech (like BERT's masked language modelling) | Predict “discrete speech units” which are analogous to tokens in a text sentence | Trained for multilingual speech recognition and hence has relatively large CTC head|

## Seq2Seq Architecture
An encoder-decoder architecture is refered to as a sequence-to-sequence model.

### In ASR (Whisper)
* Input: A log-mel spectrogram (from audio) goes into the transformer encoder, which produces a sequence of hidden states that capture the meaning of the input speech.
* Decoder: A transformer decoder takes over using:
    * Cross-attention to reference encoder outputs.
    * Causal self-attention to ensure tokens are generated in order (can’t see future).
* Autoregressive Prediction: The decoder starts from a start token (SOT) and generates one text token at a time, feeding previous outputs back in until it hits an end token or reaches a limit.

The seq2seq model offers greater flexibility and superior performance as compared to CTC using the same training data and loss function. 

### Text-to-Speech (SpeechT5)
* Input: A sequence of text tokens is passed to the transformer encoder, which extracts hidden-states that represent the text's meaning.
* Decoder: The transformer decoder uses:
    * Cross-attention to focus on encoder outputs.
    * Starts with a zero-length spectrogram (like a “start” token).
    * Autoregressively predicts the next spectrogram slice at each step.
* Stopping Criterion:
    * The decoder also predicts a "stop probability" at each step.
    * If this probability exceeds a threshold (e.g., 0.5), generation stops.
* Post-Processing:
    * A post-net (CNN layers) refines the predicted spectrogram.
    * A separate vocoder converts the final spectrogram into audio waveform.
* Loss Function: Typically L1 loss or MSE between predicted and target spectrograms
* Challenges: TTS is a one-to-many problem - the same text can be spoken in many valid ways (different pitch, rhythm, emphasis), making L1 or MSE meaningless. Insead **human listeners using MOS (Mean Opinion Score)** is often used.

## Audio Classification
### Spectrogram as images
* A spectrogram is a 2D tensor: (frequencies, time).
* It can be treated like an image and fed into models like ResNet or Vision Transformers (ViT).
* Audio Spectrogram Transformer (AST) is a ViT-like model that:
    * Splits the spectrogram into 16×16 patches.
    * Converts patches into embeddings.
    * Feeds them into a transformer encoder.
    * Uses a sigmoid classification head to output probabilities.

Note: Unlike images, shifting a spectrogram vertically changes frequency → changes meaning. So spectrograms ≠ images exactly.

### Using Transformers
* Encoder-only models (like Wav2Vec2) can be repurposed for classification tasks:
    * Wav2Vec2ForCTC → for speech recognition.
    * Wav2Vec2ForSequenceClassification → for classifying entire audio clips.   
        * Take the mean of hidden-states over time.
        * Feed it into a classification layer → one prediction per audio sample
    * Wav2Vec2ForAudioFrameClassification → for classifying each frame (e.g., for emotion or phoneme detection).
        * Apply classifier to each hidden-state.
        * Output is a sequence of predictions, one per frame.