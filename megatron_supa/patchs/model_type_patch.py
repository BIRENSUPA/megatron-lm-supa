def patch_model_type():
    try:
        from megatron.core.enums import ModelType as core_model_type
        from megatron.core.transformer.enums import ModelType as transformer_model_type
    except ImportError:
        return

    for cls in (core_model_type, transformer_model_type):
        if not hasattr(cls, 'encoder_and_decoder'):
            cls.encoder_and_decoder = 2
