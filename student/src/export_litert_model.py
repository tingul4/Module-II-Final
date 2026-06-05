import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEMMA4_CHAT_TEMPLATE_SOURCE = "google/gemma-4-E2B-it"
DEFAULT_GEMMA4_VISION_MAX_SOFT_TOKENS = 280


EXPORT_GUIDE_TEMPLATE = """# LiteRT-LM Export Guide

This workspace was generated for a merged Gemma model.

## Active flow
1. Convert the merged Hugging Face Gemma model to LiteRT split `.tflite` artifacts with LiteRT Torch.
2. Bundle the exported text, vision, and tokenizer assets into `model.litertlm`.

## Official references
- https://huggingface.co/litert-community
- https://ai.google.dev/edge/litert
- https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android

## Suggested conversion skeleton
```python
from litert_torch.generative.export_hf.export import export

export(
    model="{merged_model_dir}",
    output_dir="{output_dir}",
    task="image_text_to_text",
    prefill_lengths=[{prefill_seq_len}],
    cache_length={kv_cache_max_len},
    quantization_recipe="{quantize}",
    export_vision_encoder=True,
    bundle_litert_lm={bundle_litert_lm},
    trust_remote_code={trust_remote_code},
)
```

## Rebuild bundle only
If the split `.tflite` files already exist but `model.litertlm` must be rebuilt, run:

```bash
python3 student/src/export_litert_model.py \
  --merged_model_dir {merged_model_dir} \
  --output_dir {output_dir} \
  --tflite_model {base_tflite_arg}
```
"""


def build_parser(
    prog: str | None = None,
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description
        or "Export a merged Gemma student model to LiteRT artifacts and bundle a LiteRT-LM artifact.",
    )
    parser.add_argument("--merged_model_dir", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "student" / "mobile_artifacts" / "gemma4_e2b",
    )
    parser.add_argument("--tflite_model", type=Path, default=None)
    parser.add_argument("--quantize", type=str, default="dynamic_wi8_afp32")
    parser.add_argument("--prefill_seq_len", type=int, default=2048)
    parser.add_argument("--kv_cache_max_len", type=int, default=4096)
    parser.add_argument("--vision_quantize", type=str, default=None)
    parser.add_argument("--keep_temporary_files", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.set_defaults(bundle_litert_lm=True)
    parser.add_argument(
        "--bundle_litert_lm",
        dest="bundle_litert_lm",
        action="store_true",
        help="Bundle or rebuild model.litertlm after LiteRT export.",
    )
    parser.add_argument(
        "--no_bundle_litert_lm",
        dest="bundle_litert_lm",
        action="store_false",
        help="Skip LiteRT-LM bundling and only emit split .tflite assets.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None):
    return build_parser().parse_args(argv)


def resolve_tokenizer_asset(merged_model_dir: Path) -> Path:
    candidates = [
        merged_model_dir / "tokenizer.json",
        merged_model_dir / "tokenizer.model",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Expected tokenizer.json or tokenizer.model under the merged model directory."
    )


def write_recipe(args, tokenizer_asset: Path):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chat_template_override = resolve_chat_template_override(args)
    recipe = {
        "merged_model_dir": str(args.merged_model_dir),
        "output_dir": str(args.output_dir),
        "quantize": args.quantize,
        "prefill_seq_len": args.prefill_seq_len,
        "kv_cache_max_len": args.kv_cache_max_len,
        "vision_quantize": args.vision_quantize,
        "keep_temporary_files": args.keep_temporary_files,
        "trust_remote_code": args.trust_remote_code,
        "bundle_litert_lm": args.bundle_litert_lm,
        "tokenizer_asset": str(tokenizer_asset),
        "tflite_model": str(args.tflite_model) if args.tflite_model else None,
        "litert_lm_artifact": str(args.output_dir / "model.litertlm"),
        "chat_template_override": (
            str(chat_template_override) if chat_template_override else None
        ),
        "gemma4_vision_max_soft_tokens": get_extra_export_kwargs(args).get(
            "gemma4_vision_max_soft_tokens"
        ),
    }
    recipe_path = args.output_dir / "conversion_recipe.json"
    recipe_path.write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    base_tflite_arg = str(args.tflite_model) if args.tflite_model else (
        args.output_dir / "model_quantized.tflite"
    )
    guide = EXPORT_GUIDE_TEMPLATE.format(
        merged_model_dir=args.merged_model_dir,
        output_dir=args.output_dir,
        quantize=args.quantize,
        prefill_seq_len=args.prefill_seq_len,
        kv_cache_max_len=args.kv_cache_max_len,
        bundle_litert_lm=str(args.bundle_litert_lm),
        trust_remote_code=str(args.trust_remote_code),
        base_tflite_arg=base_tflite_arg,
    )
    guide_path = args.output_dir / "EXPORT_GUIDE.md"
    guide_path.write_text(guide, encoding="utf-8")
    return recipe_path, guide_path


def find_first_existing(candidates):
    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            return Path(candidate)
    return None


def get_model_config(args):
    try:
        from transformers import AutoConfig
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is not installed in the current environment; cannot inspect the merged model config."
        ) from exc
    return AutoConfig.from_pretrained(
        str(args.merged_model_dir),
        trust_remote_code=args.trust_remote_code,
    )


def resolve_chat_template_override(args):
    local_template = args.merged_model_dir / "chat_template.jinja"
    if local_template.exists():
        return local_template
    model_config = get_model_config(args)
    if getattr(model_config, "model_type", None) == "gemma4":
        return DEFAULT_GEMMA4_CHAT_TEMPLATE_SOURCE
    return None


def get_extra_export_kwargs(args):
    model_config = get_model_config(args)
    if getattr(model_config, "model_type", None) == "gemma4":
        return {
            "gemma4_vision_max_soft_tokens": DEFAULT_GEMMA4_VISION_MAX_SOFT_TOKENS,
        }
    return {}


def run_litert_export(args):
    try:
        from litert_torch.generative.export_hf.export import export
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "litert-torch is not installed in the current environment; cannot export the merged model."
        ) from exc

    extra_kwargs = get_extra_export_kwargs(args)
    chat_template_override = resolve_chat_template_override(args)
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
        jinja_chat_template_override=(
            str(chat_template_override) if chat_template_override else None
        ),
        **extra_kwargs,
    )


def resolve_export_outputs(args):
    return {
        "base_model": find_first_existing(
            [
                args.tflite_model,
                args.output_dir / "model_quantized.tflite",
                args.output_dir / "model.tflite",
            ]
        ),
        "vision_encoder": find_first_existing(
            [
                args.output_dir / "vision_encoder_quantized.tflite",
                args.output_dir / "vision_encoder.tflite",
            ]
        ),
        "vision_adapter": find_first_existing(
            [
                args.output_dir / "vision_adapter_quantized.tflite",
                args.output_dir / "vision_adapter.tflite",
            ]
        ),
        "litert_lm": find_first_existing([args.output_dir / "model.litertlm"]),
        "embedder": find_first_existing(
            [
                args.output_dir / "embedder_quantized.tflite",
                args.output_dir / "embedder.tflite",
            ]
        ),
        "per_layer_embedder": find_first_existing(
            [
                args.output_dir / "per_layer_embedder_quantized.tflite",
                args.output_dir / "per_layer_embedder.tflite",
            ]
        ),
        "tokenizer_json": find_first_existing(
            [
                args.output_dir / "tokenizer.json",
                args.merged_model_dir / "tokenizer.json",
            ]
        ),
    }


def maybe_package_litert_lm(args, outputs):
    if not args.bundle_litert_lm:
        return outputs["litert_lm"]
    if outputs["base_model"] is None or outputs["tokenizer_json"] is None:
        return None
    try:
        from litert_torch.generative.export_hf.core import (
            export_lib,
            exportable_module_config,
            litert_lm_builder,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "litert-torch is not installed in the current environment; cannot package a LiteRT-LM artifact."
        ) from exc

    extra_kwargs = get_extra_export_kwargs(args)
    chat_template_override = resolve_chat_template_override(args)
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
        jinja_chat_template_override=(
            str(chat_template_override) if chat_template_override else None
        ),
        extra_kwargs=extra_kwargs,
    )
    additional_model_paths = {}
    if outputs["per_layer_embedder"] is not None:
        additional_model_paths["per_layer_embedder"] = str(
            outputs["per_layer_embedder"]
        )
    exported = export_lib.ExportedModelArtifacts(
        prefill_decode_model_path=str(outputs["base_model"]),
        embedder_model_path=(
            str(outputs["embedder"]) if outputs["embedder"] is not None else None
        ),
        vision_encoder_model_path=(
            str(outputs["vision_encoder"])
            if outputs["vision_encoder"] is not None
            else None
        ),
        vision_adapter_model_path=(
            str(outputs["vision_adapter"])
            if outputs["vision_adapter"] is not None
            else None
        ),
        tokenizer_model_path=str(outputs["tokenizer_json"]),
        additional_model_paths=additional_model_paths or None,
    )
    packaged = litert_lm_builder.package_model(source, config, exported)
    if packaged.litert_lm_model_path:
        return Path(packaged.litert_lm_model_path)
    return None


def validate_outputs(args, outputs):
    if args.dry_run:
        return
    if outputs["base_model"] is None:
        raise RuntimeError(
            "LiteRT export did not produce a base model (.tflite). Check the packaging environment and export logs."
        )
    if args.bundle_litert_lm and outputs["litert_lm"] is None:
        raise RuntimeError(
            "LiteRT export completed but model.litertlm is missing after bundling."
        )


def print_summary(recipe_path: Path, guide_path: Path, outputs):
    print(f"Conversion recipe: {recipe_path}")
    print(f"Export guide: {guide_path}")
    labels = [
        ("Base LiteRT model", outputs["base_model"]),
        ("Vision encoder LiteRT model", outputs["vision_encoder"]),
        ("Vision adapter LiteRT model", outputs["vision_adapter"]),
        ("Embedder LiteRT model", outputs["embedder"]),
        ("Per-layer embedder LiteRT model", outputs["per_layer_embedder"]),
        ("LiteRT-LM artifact", outputs["litert_lm"]),
    ]
    for label, value in labels:
        if value is not None:
            print(f"{label}: {value}")


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    tokenizer_asset = resolve_tokenizer_asset(args.merged_model_dir)
    recipe_path, guide_path = write_recipe(args, tokenizer_asset)
    if not args.dry_run and args.tflite_model is None:
        run_litert_export(args)
    outputs = resolve_export_outputs(args)
    if (
        args.bundle_litert_lm
        and outputs["litert_lm"] is None
        and not args.dry_run
    ):
        outputs["litert_lm"] = maybe_package_litert_lm(args, outputs)
    validate_outputs(args, outputs)
    print_summary(recipe_path, guide_path, outputs)


if __name__ == "__main__":
    main()
