PART_TO_PATH = {
    'v': ('attn', 'W_V'),
    'k': ('attn', 'W_K'),
    'q': ('attn', 'W_Q'),
    'result': ('attn', 'W_O'),
    'pre': ('mlp', 'W_in'),
    'mlp_out': ('mlp', 'W_out'),
}

PART_TO_BIAS_PATH = {
    'v': ('attn', 'b_V'),
    'k': ('attn', 'b_K'),
    'q': ('attn', 'b_Q'),
    'result': ('attn', 'b_O'),
    'pre': ('mlp', 'b_in'),
    'mlp_out': ('mlp', 'b_out'),
}

PREACTIVATION_NAMES = {
    'q': 'resid_pre',
    'k': 'resid_pre',
    'v': 'resid_pre',
    'pre': 'resid_mid',
    'mlp_out': 'pre',
    'result': 'z',
}
