try:
    from .fitter import DlibERT, DlibWrapper
except ImportError:
    # If dlib is not installed then we shouldn't import anything into this
    # module.
    pass
