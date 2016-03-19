import ctypes


_distances = ctypes.CDLL('libdist.dylib')
_distances.md_dtw.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
_distances.md_dtw.restype = ctypes.c_double


def md_dtw(md_time_series1, md_time_series2):
    global _distances
    assert len(md_time_series1) == len(md_time_series2)
    len_ = len(md_time_series1)
    array_type = ctypes.c_double * len_
    result = _distances.md_dtw(ctypes.c_int(len_), array_type(*md_time_series1),  array_type(*md_time_series2))
    return float(result)