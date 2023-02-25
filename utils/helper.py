""" some chore utils
"""

def make_param_int(param, key_names):
    for key, _ in param.items():
        if key in key_names:
            param[key] = int(param[key])
    return param
