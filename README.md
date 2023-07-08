# pdf-to-speech

Converts a PDF to Speech

Utilizes Facebook's MMS to do the speech synthesis: https://huggingface.co/facebook/mms-tts

A lot of code copied from: https://github.com/facebookresearch/fairseq/blob/main/examples/mms/README.md#tts-1 (esp the ipynb)


How to run:
- Make sure you have python=3.10
- Don't use librosa==0.8.0 like facebook suggested, use 0.10.0 (np.complex is depricated)
- pip install -r requirements.txt

Code is pretty messy, might/(will fix later) fix (am doing this in my free time lol)

Todo list:
- Get speedup working w/o chipmunk effect
- More voices
- Try out different models - find best quality/time of inference
- UI to highlight what you're reading
