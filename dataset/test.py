import h5py

def show(name, obj):
    kind = "Group" if isinstance(obj, h5py.Group) else "Dataset"
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    print(f"{kind}: {name} | shape={shape} | dtype={dtype}")

with h5py.File("positive_proteins.h5", "r") as f:
    f.visititems(show)