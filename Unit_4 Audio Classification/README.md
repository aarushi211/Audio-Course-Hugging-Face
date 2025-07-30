# Audio Classification
### Keyword Spotting (KWS)
* Identifies the keyword in a spoken utterance. 
* The set of possible keywords forms the set of predicted class labels. 
Hence, to use a pre-trained keyword spotting model, you should ensure that your keywords match those that the model was pre-trained on.

### Speech Commands
It is a dataset containing 15 command words, a silence class, and an unknown class (for false positives)<br>Examples: "yes", "no", "stop", "go", "backward".
<br>Used to evaluate audio classification models on simple spoken commands.

**Real Word Use**
* Models like this run on phones to detect wake words (e.g., “Hey Siri”).
* Audio classifier is lightweight (millions of parameters) and always on.
* Triggers larger speech recognition models (hundreds of millions of parameters) only when needed making it battery-efficient.

### Language Identification (LID)
* It is the task of identifying the language spoken in an audio sample from a list of candidate languages.
* LID can form an important part in many speech pipelines. 
* For example, given an audio sample in an unknown language, an LID model can be used to categorise the language(s) spoken in the audio sample, and then select an appropriate speech recognition model trained on that language to transcribe the audio.

### FLEURS
FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) is a dataset for evaluating speech recognition systems in 102 languages, including many that are classified as ‘low-resource’.

### Zero-Shot Audio Classification
* Enables classification of unseen classes without retraining.
* Done using the CLAP model (Contrastive Language-Audio Pretraining), which takes:
    * Audio input
    * Candidate text labels
* It returns similarity scores between the audio and each text label.

Note: a-priori is a set of possible labels in our classification problem. Smaller label sets may yield better accuracy but less coverage.

* CLAP is trained on environmental sounds, not well-suited for fine-grained tasks like language identification (LID).
* For tasks like LID, specialized models trained on speech data are still better.