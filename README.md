# pdf-to-speech

Converts a PDF to Speech

Utilizes Facebook's MMS to do the speech synthesis: https://huggingface.co/facebook/mms-tts

A lot of code copied from: https://github.com/facebookresearch/fairseq/blob/main/examples/mms/README.md#tts-1 (esp the ipynb)


How to run:
- Make sure you have python=3.10
- Make sure to git clone the vits (use git clone --recurse-submodules https://github.com/DicksonWu654/pdf-to-speech.git, or just git clone vits)
  - You then have to go inside of vits, run `cd monotonic_align`, run `mkdir monotonic_align`, finally run: `python setup.py build_ext --inplace`
  - It's a limition of Facebook's mms :(
- pip install -r requirements.txt
  - Don't use librosa==0.8.0 like facebook suggested, use 0.10.0 (np.complex is depricated)

Code is pretty messy, will fix later 

Todo list:
- Get speedup working w/o chipmunk effect
- More voices
- Try out different models - find best quality/time of inference
- UI to highlight what you're reading
