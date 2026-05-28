import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


EXPORT_GUIDE_TEMPLATE = """# MediaPipe Task Export Guide

This workspace was generated for a merged Gemma model.

## Official flow
1. Convert the merged Hugging Face Gemma model to a LiteRT `.tflite` model with LiteRT Torch.
2. Bundle the LiteRT model with tokenizer assets into a MediaPipe `.task` file.

## Official references
- https://ai.google.dev/gemma/docs/conversions/hf-to-mediapipe-task
- https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android

## Suggested conversion skeleton
```python
from litert_torch.generative.utilities import converter
from litert_torch.generative.utilities.export_config import ExportConfig
from litert_torch.generative.layers import kv_cache

# Replace this builder with the Gemma 4 E2B builder supported by your installed litert-torch release.
# Example from the current public guide uses gemma3 builders.

export_config = ExportConfig()
export_config.kvcache_layout = kv_cache.KV_LAYOUT_TRANSPOSED
export_config.mask_as_input = True

converter.convert_to_tflite(
    pytorch_model,
    output_path="{output_dir}",
    output_name_prefix="{model_prefix}",
    prefill_seq_len={prefill_seq_len},
    kv_cache_max_len={kv_cache_max_len},
    quantize="{quantize}",
    export_config=export_config,
)
```

## Bundle step
If you already have a `.tflite` model, run this repo script again with:

```bash
python3 student/src/export_mediapipe_task.py \\
  --merged_model_dir {merged_model_dir} \\
  --output_dir {output_dir} \\
  --tflite_model {output_dir}/{model_prefix}.tflite
```

## Tokenizer note
If MediaPipe bundling rejects `tokenizer.json` because it expects a SentencePiece
`tokenizer.model`, this repo script will still preserve the exported LiteRT files
and can package a LiteRT-LM (`model.litertlm`) fallback from the Hugging Face
tokenizer JSON.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Export a merged Gemma student model to LiteRT artifacts and optionally bundle a MediaPipe .task.")
    parser.add_argument("--merged_model_dir", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "student" / "mobile_artifacts" / "gemma4_e2b",
    )
    parser.add_argument("--tflite_model", type=Path, default=None)
    parser.add_argument("--tokenizer_model_path", type=Path, default=None)
    parser.add_argument("--task_name", type=str, default="gemma4-e2b-authenticity")
    parser.add_argument("--quantize", type=str, default="dynamic_wi8_afp32")
    parser.add_argument("--prefill_seq_len", type=int, default=2048)
    parser.add_argument("--kv_cache_max_len", type=int, default=4096)
    parser.add_argument("--vision_quantize", type=str, default=None)
    parser.add_argument("--keep_temporary_files", action="store_true")
    parser.add_argument("--bundle_litert_lm", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--start_token", type=str, default="<bos>")
    parser.add_argument("--stop_tokens", type=str, nargs="+", default=["<eos>", "<end_of_turn>"])
    parser.add_argument("--prompt_prefix_user", type=str, default="<start_of_turn>user\n")
    parser.add_argument("--prompt_suffix_user", type=str, default="<end_of_turn>\n")
    parser.add_argument("--prompt_prefix_model", type=str, default="<start_of_turn>model\n")
    parser.add_argument("--prompt_suffix_model", type=str, default="<end_of_turn>\n")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_tokenizer_model(merged_model_dir: Path, tokenizer_model_path: Path | None = None) -> Path:
    if tokenizer_model_path is not None:
        if tokenizer_model_path.exists():
            return tokenizer_model_path
        raise FileNotFoundError(f"Provided tokenizer model path does not exist: {tokenizer_model_path}")
    candidates = [
        merged_model_dir / "tokenizer.model",
        merged_model_dir / "tokenizer.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Expected tokenizer.model or tokenizer.json under the merged model directory.")


def write_recipe(args, tokenizer_model: Path):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recipe = {
        "merged_model_dir": str(args.merged_model_dir),
        "output_dir": str(args.output_dir),
        "task_name": args.task_name,
        "quantize": args.quantize,
        "prefill_seq_len": args.prefill_seq_len,
        "kv_cache_max_len": args.kv_cache_max_len,
        "vision_quantize": args.vision_quantize,
        "keep_temporary_files": args.keep_temporary_files,
        "bundle_litert_lm": args.bundle_litert_lm,
        "tokenizer_model": str(tokenizer_model),
        "tflite_model": str(args.tflite_model) if args.tflite_model else None,
        "task_bundle": str(args.output_dir / f"{args.task_name}.task"),
    }
    recipe_path = args.output_dir / "conversion_recipe.json"
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")

    guide = EXPORT_GUIDE_TEMPLATE.format(
        merged_model_dir=args.merged_model_dir,
        output_dir=args.output_dir,
        model_prefix=args.task_name,
        quantize=args.quantize,
        prefill_seq_len=args.prefill_seq_len,
        kv_cache_max_len=args.kv_cache_max_len,
    )
    guide_path = args.output_dir / "EXPORT_GUIDE.md"
    guide_path.write_text(guide, encoding="utf-8")
    return recipe_path, guide_path


def find_first_existing(candidates):
    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            return Path(candidate)
    return None


def run_litert_export(args):
    try:
        from litert_torch.generative.export_hf.export import export
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "litert-torch is not installed in the current environment; cannot export the merged model."
        ) from exc

    export(
        model=str(args.merged_model_dir),
        output_dir=str(args.output_dir),
        task="image_text_to_text",
        keep_temporary_files=args.keep_temporary_files,
        trust_remote_code=args.trust_remote_code,
        prefill_lengths=[args.prefill_seq_len],
        cache_length=args.kv_cache_max_len,
        quantization_recipe=args.quantize,
        export_vision_encoder=True,
        vision_encoder_quantization_recipe=args.vision_quantize or args.quantize,
        bundle_litert_lm=args.bundle_litert_lm,
    )


def resolve_export_outputs(args):
    base_model = find_first_existing(
        [
            args.tflite_model,
            args.output_dir / "model_quantized.tflite",
            args.output_dir / "model.tflite",
        ]
    )
    vision_encoder = find_first_existing(
        [
            args.output_dir / "vision_encoder_quantized.tflite",
            args.output_dir / "vision_encoder.tflite",
        ]
    )
    vision_adapter = find_first_existing(
        [
            args.output_dir / "vision_adapter_quantized.tflite",
            args.output_dir / "vision_adapter.tflite",
        ]
    )
    litert_lm = find_first_existing(
        [
            args.output_dir / "model.litertlm",
            args.output_dir / f"{args.task_name}.litertlm",
        ]
    )
    embedder = find_first_existing(
        [
            args.output_dir / "embedder_quantized.tflite",
            args.output_dir / "embedder.tflite",
        ]
    )
    per_layer_embedder = find_first_existing(
        [
            args.output_dir / "per_layer_embedder_quantized.tflite",
            args.output_dir / "per_layer_embedder.tflite",
        ]
    )
    tokenizer_json = find_first_existing(
        [
            args.output_dir / "tokenizer.json",
            args.merged_model_dir / "tokenizer.json",
        ]
    )
    return base_model, vision_encoder, vision_adapter, litert_lm, embedder, per_layer_embedder, tokenizer_json


def maybe_package_litert_lm(args, base_model: Path | None, vision_encoder: Path | None, vision_adapter: Path | None, embedder: Path | None, per_layer_embedder: Path | None, tokenizer_json: Path | None):
    if base_model is None or tokenizer_json is None:
        return None
    try:
        from litert_torch.generative.export_hf.core import export_lib, exportable_module_config, litert_lm_builder
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "litert-torch is not installed in the current environment; cannot package a LiteRT-LM artifact."
        ) from exc

    source = export_lib.load_model(
        str(args.merged_model_dir),
        trust_remote_code=args.trust_remote_code,
        task="image_text_to_text",
    )
    config = exportable_module_config.ExportableModuleConfig(
        model=str(args.merged_model_dir),
        output_dir=str(args.output_dir),
        work_dir=str(args.output_dir),
        task="image_text_to_text",
        trust_remote_code=args.trust_remote_code,
        prefill_lengths=[args.prefill_seq_len],
        cache_length=args.kv_cache_max_len,
        quantization_recipe=args.quantize,
        export_vision_encoder=True,
        vision_encoder_quantization_recipe=args.vision_quantize or args.quantize,
        bundle_litert_lm=True,
    )
    additional_model_paths = {}
    if per_layer_embedder is not None:
        additional_model_paths["per_layer_embedder"] = str(per_layer_embedder)
    exported = export_lib.ExportedModelArtifacts(
        prefill_decode_model_path=str(base_model),
        embedder_model_path=str(embedder) if embedder else None,
        vision_encoder_model_path=str(vision_encoder) if vision_encoder else None,
        vision_adapter_model_path=str(vision_adapter) if vision_adapter else None,
        tokenizer_model_path=str(tokenizer_json),
        additional_model_paths=additional_model_paths or None,
    )
    packaged = litert_lm_builder.package_model(source, config, exported)
    return Path(packaged.litert_lm_model_path) if packaged.litert_lm_model_path else None


def maybe_bundle_task(args, tokenizer_model: Path, base_model: Path | None, vision_encoder: Path | None, vision_adapter: Path | None):
    if base_model is None:
        return None
    try:
        from mediapipe.tasks.python.genai import bundler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "mediapipe is not installed in the current environment; cannot bundle a .task artifact."
        ) from exc

    output_filename = args.output_dir / f"{args.task_name}.task"
    config = bundler.BundleConfig(
        tflite_model=str(base_model),
        tokenizer_model=str(tokenizer_model),
        start_token=args.start_token,
        stop_tokens=args.stop_tokens,
        output_filename=str(output_filename),
        prompt_prefix_user=args.prompt_prefix_user,
        prompt_suffix_user=args.prompt_suffix_user,
        prompt_prefix_model=args.prompt_prefix_model,
        prompt_suffix_model=args.prompt_suffix_model,
        tflite_vision_encoder=str(vision_encoder) if vision_encoder else None,
        tflite_vision_adapter=str(vision_adapter) if vision_adapter else None,
    )
    if not args.dry_run:
        bundler.create_bundle(config)
    return output_filename


def main():
    args = parse_args()
    tokenizer_model = resolve_tokenizer_model(args.merged_model_dir, args.tokenizer_model_path)
    recipe_path, guide_path = write_recipe(args, tokenizer_model)
    if not args.dry_run and args.tflite_model is None:
        run_litert_export(args)
    base_model, vision_encoder, vision_adapter, litert_lm, embedder, per_layer_embedder, tokenizer_json = resolve_export_outputs(args)
    task_path = None
    bundling_error = None
    try:
        task_path = maybe_bundle_task(args, tokenizer_model, base_model, vision_encoder, vision_adapter)
    except (RuntimeError, ValueError) as exc:
        bundling_error = exc
        if litert_lm is None and not args.dry_run:
            litert_lm = maybe_package_litert_lm(
                args,
                base_model,
                vision_encoder,
                vision_adapter,
                embedder,
                per_layer_embedder,
                tokenizer_json,
            )
    print(f"Conversion recipe: {recipe_path}")
    print(f"Export guide: {guide_path}")
    if base_model is not None:
        print(f"Base LiteRT model: {base_model}")
    else:
        print("Base LiteRT model: not found")
    if vision_encoder is not None:
        print(f"Vision encoder LiteRT model: {vision_encoder}")
    if vision_adapter is not None:
        print(f"Vision adapter LiteRT model: {vision_adapter}")
    if litert_lm is not None:
        print(f"LiteRT-LM artifact: {litert_lm}")
    if bundling_error is not None:
        print(f"MediaPipe task bundling skipped: {bundling_error}")
    if task_path is not None:
        action = "Planned" if args.dry_run else "Created"
        print(f"{action} MediaPipe task bundle: {task_path}")
    else:
        print("No MediaPipe .task bundle was created.")


if __name__ == "__main__":
    main()
