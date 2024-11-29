import os
import json
import argparse
from PIL import Image

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from llava.demo_utils import infer_image
from llava.model import LlavaLlamaForCausalLM


def main(image_path, questions):
    model_path = "4bit/llava-v1.5-13b-3GB"
    kwargs = {"device_map": "auto"}
    kwargs['load_in_4bit'] = True
    kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda')

    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    image_tensor = vision_tower.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    json_dict = {}
    for question in questions:
        print(f"Query : {question}")
        output = infer_image(image_tensor, question, model, tokenizer)
        print("Response :", output.strip('</s>'))
        print("\n")
        json_dict[question] = output.strip('</s>')
    
    final_dict = {
                    "filename" : os.path.basename(image_path),
                	"base info" : {"width", w, "height", h},
                    "image tags" : json_dict,
                    "object attributes" : "NA", 
	                "image descrption" : " ",
                }
    json_object = json.dumps(final_dict, indent=4)
    json_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    with open(json_filename, "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caption images using a pre-trained model.")
    parser.add_argument("--image_path", default = "", type=str, help="Path to the image file.")
    parser.add_argument("--prompts", nargs='+', type=str, help="List of questions for image captioning.")
    args = parser.parse_args()
    main(args.image_path, args.prompts)
