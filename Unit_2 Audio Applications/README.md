# Audio Applications
This unit will cover some of the basic applications of audio such as -
* Audio Classification
* Automatic Speech Recognition
* Text to Speech

There is another application called **Speech Diarization** which is used to differentiate between mutiple speakers in an audio. It will be covered in later unit.

## Audio Classification
As the word classification suggests, in this we are trying to classify various audio inputs into different categories such as barking dog or a meowing cat, or what music genre a song belongs to.

Just as any other classification data, we will also need the data to be labelled with its category. In the [Audio-Applications](/Unit 2/Audio Applications.ipynb), we are using [MINDS-14](https://huggingface.co/datasets/PolyAI/minds14) dataset which contains intent_class for each audio. In this case we can simply use a pipeline and a pre-trained model to perform classification. The model used is fine-tuned on MINDS-14 dataset. 

All the pre-processing of the dataset can easily be handled by the pipeline itself. Hence, we just need to enter the input in the classifier and it will return the confidence score. 

## Automatic Speech Recognition
In this, we transform audio clips to text automatically. ASR is commonly used to create captions for videos and even commands for virtual assistants like Siri and Alexa.

Similar to how Audio Classification was performed using the pipeline, we can perform ASR using pipeline too.

> Benifits of working with a pipeline
    >* a pre-trained model may exist that already solves your task really well, saving you plenty of time
    >* pipeline() takes care of all the pre/post-processing for you, so you don’t have to worry about getting the data into the right format for a model
    >* if the result isn’t ideal, this still gives you a quick baseline for future fine-tuning

## Audio Generation
Audio or speech generation AKA Text-to-Speech transforms text to lifelike spoken language sound. This can be used in virtual assistants, accessibility tools for the visually impaired, and personalized audiobooks.

This can be further improved for music generation can enable creative expression and can be used in entertainment and game development industries.

In a similar manner to the above applications, a pipeline can be used for this task.