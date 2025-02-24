# SGDImagePrompt
A Training-free method for prompting Stable Diffusion with an image, leveraging the fact that our text encoder is already aligned
with image distribution via CLIP training.
With this insight we can run gradient descent on the prompt embeddings, using its alignment with the CLIP image embedding as a loss.

Also check out the related [QuickEmbedding](https://github.com/ethansmith2000/QuickEmbedding/tree/master)

# Setup 🛠
```
pip install -r requirements.txt
```


# Inference 🚀
See the provided notebook

# Settings
The opimize_chunk function will train an additional set of tokens to append to the end of your prompt. 
Meanwhile, optimize_prompt will replace your prompt.
- anchor_factor: higher values will try to better preserve the original prompt
- use_negative_prompt: whether to also optimize the negative prompt
- steps: optimization steps
- lr: learning rate
- optimizer: optimizer to use
- actual_eof: which token to use for CLIP loss, normally it is one token position beyond the length of your prompt, if False, it will be the final token
- clip_num: For SDXL, signals whether we're using the vision encoder for vitl14 or the open-clip vitg
- lr_penalty: amount to reduce lr if we find that the loss is increasing
- modality_gap_embed: an attempt to close CLIP's modality gap, this involves creating a mean of CLIP image embeddings and CLIP text embeddings, the difference between these two is what would be provided
- isolate_subject: isolate the subject via attention masking, mask estimated by rembg background removal

