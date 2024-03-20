COMPONENTS = ['q', 'k', 'v', 'pre', 'mlp_out', 'result']

COMPONENT_TO_OBJECT_PATH = {
    'v': ('attn', 'W_V'),
    'k': ('attn', 'W_K'),
    'q': ('attn', 'W_Q'),
    'result': ('attn', 'W_O'),
    'pre': ('mlp', 'W_in'),
    'mlp_out': ('mlp', 'W_out'),
}

PREACTIVATION_NAMES = {
    'q': 'resid_pre',
    'k': 'resid_pre',
    'v': 'resid_pre',
    'pre': 'resid_mid',
    'mlp_out': 'pre',
    'result': 'z',
}
