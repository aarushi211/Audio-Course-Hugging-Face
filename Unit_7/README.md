# Speech to Speech Translation (STST)
There are two approaches we can use to perform STST, that is translating one language speech to another language.

1. **Cascaded Approach -**<br>
This approach uses 2 steps
    * Speech Translation: To transcribe source speech into text in the target language
    * TTS: Generate speech in target language from translated text

2. **Another Approach -**<br>
This contains 3 steps -
    * ASR: Transcribe speech into text of same language
    * Machine Translation: Translate transcribed text into text of target language
    * TTS: Speech in target language

Adding more components to the pipeline leads to error propagation and also increases latency since inference has to be conducted for more models. 

## Voice Assistant
The steps to create a voice assistant can be broken down into 4 stages -
### Wake word detection: 
* Voice assistants are consistently listening to the audio inputs comping through the device's microphone but only boot into action when a particular wake-up word is spoken.
* The detection is handled by small on-device audio classification which is smaller and lighter. 
* It can run on device without draining battery.
* When the wake word is detected, the larger speech recognition model is launched and afterwards shut down again.

### Speech Transcription: 
* Due to large size if audio files, transferring them to cloud is slow.
* Instead a small ASR model on-device is used to transcribe.
* Though it is less accurate, but faster inference speed makes it worthwhile since it can be used in real-time.

### Language Model Query:
* To generate responses LLM is used.
* Since the query is small and LLMs are large, the query is send to LLM on cloud to generate text responses and return back to the device.

### Synthesize Speech:
* TTS is used to synthesize the text to speech. 
* Can be done on-device but also on cloud and then transferring it to device.

## Transcribing a Meeting
It generates transcription for a conversation in a meeting with multiple speakers. To properly transcribe a meeting and segment who spoke when, the process can be broken down into 3 steps -
1. **Speaker Diarization:** It helps map who spoke when. This helps create end-to-end meeting transcription with filly formatted start and end times for each speaker. `pyannote/speaker-diarization` is commonly used for this.
2. **Speech transcription:** Now, we can get the transcriptions along with their timestamps using models like `Whisper`. Note, timestamps don’t always perfectly align with diarization outputs.
3. **Aligning Diarization & Transcription:** Since Diarization and ASR models give slightly different segment timings, we need to align segments by minimizing the absolute difference in timestamps. For this, `ASRDiarizationPipeline` can be used.
