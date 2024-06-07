# Mars 5 technical details

While we do not have the time for a proper full writeup of the details of Mars5, its design, training, and implementation, we at least try give a more detailed overview here of how Mars5 works.


## hubconf object/api


After loading the model with `torch.hub.load`, two objects are returned, a Mars5TTS, and the dataclass of the inference config to use when calling the `mars5.tts()` method.
Concretely, the main methods of the mars5 object are:

```python

# The init function, called automatically when you initialize the 
# model from torch.hub.load(). If you want, you can pass in your
# own custom checkpoints here to initalize the model with your 
# own model, tokenizer, etc...
def __init__(self, ar_ckpt, nar_ckpt, device: str = None) -> None:
    # ... initialization code ...

# Main text-to-speech function, converting text and a reference
# audio to speech. 
def tts(self, text: str, ref_audio: Tensor, ref_transcript: str | None, 
            cfg: InferenceConfig) -> Tensor:
        """ Perform TTS for `text`, given a reference audio `ref_audio` (of shape [sequence_length,], sampled at 24kHz) 
        which has an associated `ref_transcript`. Perform inference using the inference 
        config given by `cfg`, which controls the temperature, top_p, etc...
        Returns:
        - `ar_codes`: (seq_len,) long tensor of discrete coarse code outputs from the AR model.
        - `out_wav`: (T,) float output audio tensor sampled at 24kHz.
        """

# Utility function to vocode encodec tokens, if one wishes 
# to hear the raw AR model ouput by vocoding the `ar_codes` 
# returned above.
def vocode(self, tokens: Tensor) -> Tensor:
    """ Vocodes tokens of shape (seq_len, n_q) """

```
