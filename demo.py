import argparse
import os

from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, TextStreamer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni video inference")
    parser.add_argument("--model_path", required=True, help="Model directory or Hugging Face model ID")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--describe", default="Describe this video.")
    parser.add_argument("--omnidtbc", action="store_true")
    parser.add_argument("--remaining_audio", type=float, default=0.6)
    parser.add_argument("--remaining_video", type=float, default=0.3)
    parser.add_argument("--video_chunk_frames", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.02)
    args = parser.parse_args()

    if args.omnidtbc:
        from utils.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    else:
        from transformers import Qwen2_5OmniForConditionalGeneration

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    if args.omnidtbc:
        model.thinker.omnidtbc_config = {
            "remaining_audio": args.remaining_audio,
            "remaining_video": args.remaining_video,
            "video_chunk_frames": args.video_chunk_frames,
            "alpha": args.alpha,
        }
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    conversation = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                    "capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": args.video},
                {"type": "text", "text": args.describe},
            ],
        },
    ]

    use_audio_in_video = True
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    if args.omnidtbc:
        model.thinker.nframes = videos[0].shape[0]
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    ).to(model.device).to(model.dtype)

    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    print("Generating...")
    model.generate(
        **inputs,
        use_audio_in_video=use_audio_in_video,
        return_audio=False,
        temperature=1,
        streamer=streamer,
    )


if __name__ == "__main__":
    main()
