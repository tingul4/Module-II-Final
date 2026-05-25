import torch
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info


def main():
    base_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    adapter_path = "/ssd4/LPCVC2026/student/outputs/20260424_210806/checkpoint-3040"

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(base_model_name)

    print("Loading base model...")
    if "Qwen2.5" in base_model_name:
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        model_class = Qwen2VLForConditionalGeneration

    base_model = model_class.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # 準備測試圖片路徑
    image_path = "/ssd4/LPCVC2026/student/teacher_dataset_04211726/images/1_fake/code_lcm-lora-sdv1-5_train2017_000000061697.jpg"

    # 讀取 Stage 1 和 Stage 2 的 prompt
    val_stage1_path = "/ssd4/LPCVC2026/Qualcomm_tools/26LPCVC_Track3_Sample_Solution/dataset/prompts/stage1.txt"
    val_stage2_path = "/ssd4/LPCVC2026/Qualcomm_tools/26LPCVC_Track3_Sample_Solution/dataset/prompts/stage2.txt"

    print("Loading prompts...")
    with open(val_stage1_path, "r", encoding="utf-8") as f:
        # 清理多餘換行
        stage1_prompts = [line.strip().strip('"') for line in f if line.strip()]

    with open(val_stage2_path, "r", encoding="utf-8") as f:
        stage2_prompt_template = f.read().strip()

    def generate_response(prompt_text, max_new_tokens=512):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
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
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        return output_text[0]

    print("\n--- Running Stage 1 ---")
    stage1_answers = []
    for i, p in enumerate(stage1_prompts):
        print(f"Generating analysis {i + 1}...")
        ans = generate_response(p, max_new_tokens=512)
        stage1_answers.append(ans)
        print(f"Result {i + 1}:\n{ans}\n")

    print("--- Running Stage 2 ---")
    combined_answers = "\n\n".join(
        [f"Analysis {i + 1}:\n{ans}" for i, ans in enumerate(stage1_answers)]
    )
    final_stage2_prompt = f"{stage2_prompt_template}\n\nHere are the analytical answers:\n{combined_answers}"

    print("Generating final JSON...")
    final_json = generate_response(final_stage2_prompt, max_new_tokens=1024)

    print("\n--- Final Stage 2 Output ---")
    print(final_json)


if __name__ == "__main__":
    main()
