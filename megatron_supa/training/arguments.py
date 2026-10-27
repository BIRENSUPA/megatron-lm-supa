from functools import wraps


def extra_args_provider_decorator(extra_args_provider):
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_br_args(parser)
        return parser

    return wrapper


def parse_args_decorator(parse_args):
    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper


def validate_args_decorator(validate_args):
    @wraps(validate_args)
    def wrapper(args, defaults={}):
        args = validate_args(args, defaults)
        if args.enable_zero_bubble:
            assert (
                args.pipeline_model_parallel_size > 1
            ), '--enable-zero-bubble requires pipeline-model-parallel-size > 1.'
            assert (
                args.virtual_pipeline_model_parallel_size is None
            ), '--enable-zero-bubble does not support virtual pipeline parallelism yet.'
            assert not args.overlap_grad_reduce, '--enable-zero-bubble does not support --overlap-grad-reduce yet.'
            assert args.gradient_accumulation_fusion, '--enable-zero-bubble requires --gradient-accumulation-fusion.'
            assert (
                args.transformer_impl != 'transformer_engine'
            ), '--enable-zero-bubble does not support transformer_engine yet; use --transformer-impl local.'
            if args.zero_bubble_max_pending_backward != 'auto':
                args.zero_bubble_max_pending_backward = int(args.zero_bubble_max_pending_backward)
        return args

    return wrapper


def process_br_args(parser):
    parser = _add_zero_bubble_args(parser)
    return parser


def _add_zero_bubble_args(parser):
    group = parser.add_argument_group('Zero Bubble Config')
    group.add_argument(
        '--enable-zero-bubble',
        action='store_true',
        help='Use the experimental Zero Bubble pipeline runtime and schedule.',
    )
    group.add_argument(
        '--zero-bubble-max-pending-backward',
        type=str,
        default='auto',
        help='Maximum pending backward count for Zero Bubble scheduling.',
    )
    return parser
