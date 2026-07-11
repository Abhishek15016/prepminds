"""
Small compatibility shim for TRL's `SFTConfig` / `DPOConfig`.

TRL has renamed/dropped constructor arguments across recent releases (e.g.
`max_prompt_length` disappearing from `DPOConfig`, `padding_free` appearing
on `SFTConfig`). Since `requirements.txt` intentionally floats to the latest
`trl` release rather than pinning an exact version, notebooks build their
config kwargs as a plain dict and pass it through `build_config()` here,
which drops anything the installed version doesn't recognize instead of
crashing outright.
"""

import inspect


def build_config(config_cls, **kwargs):
    accepted = set(inspect.signature(config_cls.__init__).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"Note: this installed TRL version's {config_cls.__name__} doesn't accept {dropped} - skipping.")
    return config_cls(**filtered)
