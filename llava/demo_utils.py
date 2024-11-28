import torch

from llava.utils import disable_torch_init
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def infer_image(image_tensor, prompt, model, tokenizer):
  disable_torch_init()
  conv_mode = "llava_v0"
  conv = conv_templates[conv_mode].copy()
  roles = conv.roles
  inp = f"{roles[0]}: {prompt}"
  inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
  conv.append_message(conv.roles[0], inp)
  conv.append_message(conv.roles[1], None)
  raw_prompt = conv.get_prompt()
  input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
  stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
  keywords = [stop_str]
  stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
  with torch.inference_mode():
    output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2,
                                max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
  outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
  conv.messages[-1][-1] = outputs
  return outputs
