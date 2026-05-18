import ctypes as ct

lib = ct.CDLL('/oem/usr/lib/librknnmrt.so')
ctx = ct.c_uint64(0)

with open('/root/yolov8n_final.rknn','rb') as f:
    data = f.read()
buf = ct.create_string_buffer(data)
ret = lib.rknn_init(ct.byref(ctx), buf, len(data), 0, None)
print('init:', ret)

ret = lib.rknn_run(ctx, None)
print('run:', ret)

class Out(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ('want_float',  ct.c_int32),
        ('is_prealloc', ct.c_int32),
        ('index',       ct.c_uint32),
        ('buf',         ct.c_void_p),
        ('size',        ct.c_uint32),
    ]

out = Out()
out.want_float = 1
out.is_prealloc = 0
out.index = 0

ret = lib.rknn_outputs_get(ctx, 1, ct.byref(out), None)
print('outputs_get:', ret, 'size:', out.size)
if ret == 0 and out.size > 0:
    vals = (ct.c_float * 5).from_address(out.buf)
    print('values:', list(vals))
    lib.rknn_outputs_release(ctx, 1, ct.byref(out))

lib.rknn_destroy(ctx)
