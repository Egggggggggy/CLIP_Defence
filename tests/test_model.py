from src.model import build_prompt_ensemble


def test_build_prompt_ensemble() -> None:
    classes = ["cat", "dog"]
    templates = ["a photo of {}", "a sketch of {}"]
    prompts = build_prompt_ensemble(classes, templates)
    assert prompts == [
        ["a photo of cat", "a sketch of cat"],
        ["a photo of dog", "a sketch of dog"],
    ]
