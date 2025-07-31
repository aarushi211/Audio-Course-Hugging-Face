# Automatic Speech Recognition
It broadly falls into 2 categories -
1. Connectionist Temporal Classification (CTC): encoder-only models with a linear classification (CTC) head on top
2. Sequence-to-sequence (Seq2Seq): encoder-decoder models, with a cross-attention mechanism between the encoder and decoder

### CTC Model
* It is an 'acoustic-only’ model.
* The encoder takes the audio input and forms the hidden state representations
* The linear layers maps these representations to characters

**Advantage:** Needs as little as 10 mins of labelled speech data to achieve strong performance on a downstream speech recognition task.

**Disadvantage:** Prone to phonetic spelling errors

### Seq2Seq
* Encoder takes the audio input and computes the hidden state representations
* Decoder takes these representations and generates the text transcriptions. Hence, plays a role of a language model.
* The decoder has the access to the global context of audio input, it is able to 
    * Correct the spelling mistakes
    * Circumvent the issue of phonetic predictions

**Downsides -**
* Decoding is done one step at a time, making it slower
* Requires large data to reach convergence

### Long-Form Transcription and Timestamps
For long audios, that is greater than 30 s, we cannot use models such as Whisper directly. This is because -
* Even on passing an audio of greater than 30s, Whisper will truncate the audio to 30s (For audios less than 30s, they get padded with silence)
* Memory required by transformers is equivalent to twice the input length, leading to frequent out-of-memory (OOM) error.

In order to work with long audio, we can follow the following steps -
* Chunking the audio to small manageable segments
* Each segment should have a small amount of overlap with the previous segment
* This overlap helps in merging the segments to form one audio

**Advantage**
* Splitting audio into chunks enables parallel computation increasing the speed.
* Transcribing can be done independently 
* Transcription of chunks is independent of other chunks making the process stateless and parallelizable
* Stitching is done at the end and hence order doesn't matter

## Features of Speech Datasets
1. **No. of hours:** 
    * This is equivalent to the number of training examples in NLP.
    * For model to better generalise, diverse dataset containing lots of different speakers, domains and speaking styles is required.
2. **Domain:**
    * The domain from where the data is collected is should be similar to the conditions at inference time.
    * Eg. data from audiobooks are high quality with no backround noise, while data from YouTube would contain background noise and informal speech.
3. **Speaking Style:**
    * It falls into 2 main categories -
        * Narrated: read from a script
        * Spontaneous: un-scripted, conversational speech
4. **Transcription Style:**
    * For fully formatted output, we need training data with punctuation and casing.
    * If not, then unformatted data can be directly used or the punctuation and casing can be removed during pre-processing.

## Evaluation Metrics
There are 3 categories of errors -
* Substitutions (S): where we transcribe the wrong word in our prediction (“sit” instead of “sat”)
* Insertions (I): where we add an extra word in our prediction
* Deletions (D): where we remove a word in our prediction

### Word Error Rate
* The word error rate (WER) metric is the ‘de facto’ metric for speech recognition. It calculates substitutions, insertions and deletions on the word level. This means errors are annotated on a word-by-word basis.
* Spelling errors are penalised heavily, no matter how minor they are. Lower the Wer the better it is. A perfect system would have 0 WER.
* Note: There is no upper limit since WER is the ratio of errors to number of words

### Word Accuracy
It is also measured at word level
WAcc = 1 - WER

### Character Error Rate
Done at character level. 
This means we divide up our words into their individual characters, and annotate errors on a character-by-character basis

### WER vs CER in Speech System Evaluation
* WER (Word Error Rate) is more commonly used than CER (Character Error Rate) because it assesses contextual understanding, like correct verb tense usage.
* WER is stricter, but better encourages development of more intelligible and context-aware speech systems.
* CER is used when WER isn't applicable — for example, in Mandarin or Japanese, where there is no clear word boundary.
* For realistic evaluation, WER should be computed over large test sets, aggregating substitutions (S), insertions (I), deletions (D), and total words (N), not just one sentence.

### Normalization
* Training data containing punctuation and casing for tasks like transcribing meetings or dictation is referred to as orthographic.
* However, normalizing the data to remove punctuation and casing makes it easier to train and get better results. However, such prediction makes the text difficult to read.
* Whisper transcriptions are orthographic and thus ready to go.
* One method to get what we want is by training the model on orthographic transcriptions and then normalize the transcription and prediction before computing WER. 

## ASR Pipeline
It has 3 stages - 
    • Feature Extractor: Pre-processes raw audio inputs to log-mel spectrograms
    • Model: Which performs seq-seq mapping
    • Tokenizer: Post-processes predicted token to text

Whisper has feature extractor and tokenizer called `WhisperFeatureExtractor` and `WhisperTokenizer` which is wrapped in class `WhisperProcessor`. It performs audio pre-processing and text post-processing.

**Training Whisper on new language**
* Whisper is not trained on Dhivehi
* When we fine-tune it on a new language, it will leverage its knowledge on the 96 languages it is trained on
* We will take advantage of the fact that all modern languages are usually linguistically similar to at least 1 language it already knows, through which we can perform cross-lingual knowledge.

**Training and Evaluation**
    1. Define Data Collator: Takes pre-processed data and prepares PyTorch tensors
    2. Evaluation metrics: Evaluate WER
    3. Load pre-trained checkpoint: Configure it correctly for training
    4. Define training arguments

**Data Collator**
* Input features and labels are processed independently.
* Input Features:
    * Already preprocessed: padded to 30s and converted to fixed-dimension log-Mel spectrograms.
    * Only require conversion to PyTorch tensors using feature_extractor.pad(..., return_tensors="pt").
    * No additional padding is applied, as all inputs are of fixed size.
* Labels:
    * Initially unpadded.
    * Padded to the maximum sequence length in the batch using tokenizer.pad(...).
    * Padding tokens are replaced with -100 to ignore them during loss computation.
    * The special start-of-transcript token is removed from the beginning, as it is added separately during training.
