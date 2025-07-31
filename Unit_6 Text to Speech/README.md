# Text to Speech
* Text-to-Speech (TTS) is a one-to-many mapping: The same text can be spoken in multiple valid ways, varying by speaker, intonation, speed, and emphasis.
* Speaker Variability: Different people pronounce the same sentence differently, resulting in unique spectrograms and waveforms.
* Timing and Duration Learning: The model must learn appropriate phoneme, word, and sentence durations — a complex task, especially for long or intricate sentences.
* Long-Distance Dependency Challenge: Since language is sequential and contextual, the model needs to retain meaning over long sequences to generate fluent and natural speech.
* Training Data Requirements: TTS models require paired datasets of text and corresponding speech recordings.
* Data Collection Complexity: Gathering high-quality speech data across multiple speakers and speaking styles is both expensive and time-consuming.

### Characteristics of TTS dataset
* High-Quality and Diverse Audio: Recordings should be clear, natural-sounding, and free from background noise. The dataset should represent a wide variety of speech patterns, accents, languages, and emotions to support expressive and multilingual synthesis.
* Accurate Transcriptions: Each audio clip must be paired with its corresponding text transcription to enable supervised learning.
* Linguistic Diversity: The dataset should include a broad range of sentence structures, phrases, and vocabulary across various topics, genres, and domains. This helps the model generalize to different linguistic contexts and speaking styles.

## Pre-trained Models
### SpeechT5
* Can be used for TTS, STT, (ASR or speaker identification), STS (speech enhancement or converting between different voices)
* Uses a regular transformer encoder-decoder model. It models a seq-to-seq transformation using hidden representations. 
* The transformer backbone is same for all tasks. 
* The transformer is complemented with 6 modal specific (speech/text) pre-nets and post-nets. 
* The input speech or text (depending on the task) is preprocessed through a corresponding pre-net to obtain the hidden representations that Transformer can use.
* The Transformer’s output is then passed to a post-net that will use it to generate the output in the target modality.

The pre- and post-nets for TTS are -
* Text Encoder Pre-Net: Converts text tokens into hidden representations using an embedding layer. This process is similar to the text embedding layers used in NLP models like BERT.
* Speech Decoder Pre-Net: Processes the input log-Mel spectrogram using a series of linear layers to compress it into hidden representations suitable for decoding.
* Speech Decoder Post-Net: Applies a refinement step by predicting a residual that is added to the output spectrogram, improving the quality and accuracy of the generated speech.

### Speaker Embeddings
* Represents speaker's identity in a compact way as vector of fixed size, regardless of length of utterance.
* They capture speaker's
    * Voice
    * Accent
    * Intonations
    * And other unique characteristics
* These embeddings are used for 
    * Speaker verification
    * Diarization
    * Identification and more

**Method 1: I-Vector (Identity Vectors)**
* Based on Gaussian Mixture Model (GMM)
* Represents speaker as low-dimensional fixed-length vectors derived from statistics of speaker specific GMM 
* Obtained in unsupervised manner

**Method 2: X-Vectors**
* DNN are used to obtain these.
* Trains to discriminate between speakers and maps variable length utterances to fixed dimensional embeddings.
* Can also load the X-Vector speaker embedding that has already been computed, which will encapsulate the speaker's characteristics.

### Vocoder based on HiFi-GAN
* Designed for high-fidelity speech synthesis.
* Generates high quality and realistic audio waveforms from spectrogram inputs.
* Consists 1 generator and 2 discriminators.
* Generator is CNN that takes mel-spectrogram as input and produces raw audio waveforms
* Discriminator distinguishes between real and generated audio.
* 2 discriminator focus on diff aspect of the audio
* HiFi-GAN uses adversarial training, where generator and discriminator networks compete against each other.
* Initially, the generator produces low-quality audio, and the discriminator can easily differentiate it from real audio. 
* As training progresses, the generator improves its output, aiming to fool the discriminator. 
* The discriminator, in turn, becomes more accurate in distinguishing real and generated audio.
* This adversarial feedback loop helps both networks improve over time. Ultimately, HiFi-GAN learns to generate high-fidelity audio that closely resembles the characteristics of the training data.

### Bark
* Bark generates raw speech waveforms directly, eliminating the need for a separate vocoder during inference.
* **Encodec**
    * Bark utilizes EnCodec, which functions as both a codec and a compression mechanism.
    * EnCodec compresses audio into a compact representation to reduce memory usage and then decompresses it to reconstruct the waveform.
    * It uses 8 discrete codebooks, each containing integer vectors that serve as embeddings of the audio.
    * Each successive codebook contributes to improving the quality of the audio reconstruction.
* Bark is made up of 4 main models -
    * **BarkSemanticModel (also known as the text model):** A causal autoregressive transformer that takes tokenized text as input and generates semantic tokens representing the meaning and intent of the text.
    * **BarkCoarseModel (the coarse acoustic model):** Another causal autoregressive transformer that processes the output from the semantic model to predict the first two audio codebooks required by the EnCodec decoder.
    * **BarkFineModel (the fine acoustic model):** A non-causal autoencoder transformer that refines the acoustic output by predicting the remaining audio codebooks, based on the cumulative embeddings of previously predicted codebooks.
    * **EnCodecModel (Decoder):** After all codebook channels are predicted, EnCodec is used to decode the full audio array, converting the discrete tokens into a playable waveform.
* The first three models can be conditioned using speaker embeddings, enabling Bark to synthesize speech in specific, predefined voices.
* Processor Component -
    * Tokenizes input text into smaller units for the model to process.
    * Stores and manages voice presets (speaker embeddings) to control the voice and speaking style.
* Key Functionalities -
    * Multilingual Support: Bark can generate speech in multiple languages. You don’t need to explicitly specify the language; just provide text in the desired language.
    * Non-verbal Audio Generation: Bark can produce non-speech audio like laughter, sighs, or crying by adding cues in the text (e.g., [laughs], [clears throat]).
    * Music Generation: Bark can also generate musical tones. Wrap your lyrics or musical text in ♪ musical notes ♪ for better results.
    * Batch Processing Support: Multiple text inputs can be processed simultaneously, although this requires significant computational resources.

### Massive Multilingual Speech (MMS)
* Supports over 1,100 languages
* Converts input text directly into raw speech waveforms, eliminating the need for a separate vocoder.
* Model Architecture:
    * Operates as a conditional variational autoencoder (VAE), estimating audio features from textual input.
    * Initially, it generates acoustic features in the form of spectrograms.
    * These features are then transformed into waveforms using transposed convolutional layers, adapted from the HiFi-GAN architecture.
* Inference Process:
    * Text encodings are upsampled and passed through a flow module.
    * The output is then fed into the HiFi-GAN decoder to generate the final waveform.
* Similar to Bark, MMS bypasses traditional vocoders by directly synthesizing high-quality speech waveforms.

## Evaluating TTS
During training, TTS models typically minimize Mean Squared Error (MSE) or Mean Absolute Error (MAE) between the predicted and target spectrograms. These loss functions encourage the model to closely match the acoustic features of the ground truth.
* **Challenge in Evaluation:** Since TTS is a one-to-many mapping problem—where the same text can be spoken in multiple valid ways—standard loss metrics often fail to fully capture the quality of generated speech.
* As a result, TTS systems rely heavily on human perception for evaluation. A widely used method is the Mean Opinion Score (MOS). In this participants listen to generated samples and rate them on a scale of 1 to 5, based on factors like naturalness and clarity.

**Why Not Use Objective Metrics?**<br>
Developing a universal objective metric is difficult due to the following reasons:
* Human speech perception is highly subjective, with individual preferences for pronunciation, intonation, clarity, and emotional expression.
* It’s hard to reduce these complex perceptual qualities into a single, consistent numerical score.
* Subjectivity also complicates benchmarking, as different listeners may rate the same audio differently.
* Some critical attributes—like naturalness, expressiveness, and the ability to evoke emotions—are inherently hard to quantify but vital for real-world applications.