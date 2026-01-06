import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with GeoReason")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--image_path", type=str, required=True, help="Root directory for images")
    parser.add_argument("--output_path", type=str, default="results.json", help="Path to save inference results")
    return parser.parse_args()

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def extract_content(text):
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""
    answer_content = answer_match.group(1).strip() if answer_match else text.strip()
    return think_content, answer_content

def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = load_dataset(args.dataset)
    results = []
    print("Starting inference...")
    for item in tqdm(dataset):

        file_id = item['file_id']
        ques_type = item['type']
        img_path = item['image_path']
        question = item['question']
        answer = item['answer']

        full_image_path = os.path.join(args.image_path, img_path)
        
        if not os.path.exists(full_image_path):
            print(f"Warning: Image not found at {full_image_path}")
            continue
        
        new_question = f"{question}\nPlease provide reasoning in <think> tags, and the correct option letter (A, B, C, or D) in <answer> tags. Please reason step by step. Format strictly as <think>...</think><answer>...</answer>"

        messages = [
            {
                "role": "system", 
                "content": "You are an expert in the field of remote sensing. You can analyze problems by thinking and finally provide accurate answers."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": full_image_path,
                    },
                    {"type": "text", "text": new_question},
                ],
            }
        ]


        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(model.device)


        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,  
                do_sample=False      
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        pre_think, pre_answer = extract_content(output_text)

        print(f"Question: {question}  Answer: {pre_answer}")

        result_item = {
            "file_id": file_id,
            "type": ques_type,
            "question": question,
            "answer": answer,
            "image_path": img_path,
            "pre_think": pre_think,
            "pre_answer": pre_answer
        }
        results.append(result_item)

    output_file = args.output_path
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Inference completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()
