"""Read COLMAP cameras, poses, and point tracks without a native dependency."""

from pathlib import Path
import struct
import numpy as np

# COLMAP model IDs and parameter counts.
MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def rotation(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def read(path):
    path = Path(path)
    cameras, images, points = {}, [], {}
    if (path / "cameras.bin").is_file():

        def unpack(stream, pattern):
            return struct.unpack(
                "<" + pattern, stream.read(struct.calcsize("<" + pattern))
            )

        with (path / "cameras.bin").open("rb") as stream:
            for _ in range(unpack(stream, "Q")[0]):
                ident, model, width, height = unpack(stream, "iiQQ")
                name, count = MODELS[model]
                cameras[ident] = dict(
                    model=name,
                    width=width,
                    height=height,
                    params=unpack(stream, "d" * count),
                )
        with (path / "images.bin").open("rb") as stream:
            for _ in range(unpack(stream, "Q")[0]):
                row = unpack(stream, "idddddddi")
                name = bytearray()
                while True:
                    char = stream.read(1)
                    if not char:
                        raise ValueError("Truncated COLMAP image name")
                    if char == b"\0":
                        break
                    name.extend(char)
                count = unpack(stream, "Q")[0]
                observations = np.frombuffer(
                    stream.read(count * 24), dtype=[("xy", "<f8", (2,)), ("id", "<i8")]
                )
                images.append(
                    dict(
                        id=row[0],
                        q=row[1:5],
                        t=row[5:8],
                        camera=row[8],
                        name=name.decode(),
                        point_ids=observations["id"].tolist(),
                    )
                )
        if (path / "points3D.bin").is_file():
            with (path / "points3D.bin").open("rb") as stream:
                for _ in range(unpack(stream, "Q")[0]):
                    row = unpack(stream, "QdddBBBd")
                    count = unpack(stream, "Q")[0]
                    stream.seek(count * 8, 1)
                    points[row[0]] = dict(xyz=list(row[1:4]), color=list(row[4:7]))
    else:

        def lines(name):
            return [
                line
                for line in (path / name).read_text().splitlines()
                if not line.startswith("#")
            ]

        for line in lines("cameras.txt"):
            if not line.strip():
                continue
            ident, model, width, height, *parameters = line.split()
            cameras[int(ident)] = dict(
                model=model,
                width=int(width),
                height=int(height),
                params=list(map(float, parameters)),
            )
        rows = lines("images.txt")
        i = 0
        while i < len(rows):
            if not rows[i].strip():
                i += 1
                continue
            row = rows[i].split(maxsplit=9)
            i += 1
            observations = rows[i].split() if i < len(rows) else []
            i += 1
            images.append(
                dict(
                    id=int(row[0]),
                    q=list(map(float, row[1:5])),
                    t=list(map(float, row[5:8])),
                    camera=int(row[8]),
                    name=row[9],
                    point_ids=[
                        int(observations[j]) for j in range(2, len(observations), 3)
                    ],
                )
            )
        if (path / "points3D.txt").is_file():
            for line in lines("points3D.txt"):
                if not line.strip():
                    continue
                row = line.split()
                points[int(row[0])] = dict(
                    xyz=list(map(float, row[1:4])), color=list(map(int, row[4:7]))
                )
    if not cameras or len(images) < 2:
        raise ValueError("COLMAP reconstruction needs at least two registered images")
    return cameras, sorted(images, key=lambda item: item["name"]), points


def intrinsics(camera):
    parameters = camera["params"]
    model = camera["model"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy, *distortion = parameters
        fx = fy = f
    elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
        fx, fy, cx, cy, *distortion = parameters
    else:
        raise ValueError(
            f"Unsupported camera model {model}; export pinhole/undistorted COLMAP images first"
        )
    return np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1.0]]), distortion


def undistort(image, camera):
    """Map ideal pinhole pixel centers to source pixels; preserve the original separately."""
    K, distortion = intrinsics(camera)
    if not distortion or not any(distortion):
        return image, K
    width, height = image.size
    y, x = np.mgrid[:height, :width]
    x = (x - K[0, 2]) / K[0, 0]
    y = (y - K[1, 2]) / K[1, 1]
    r2 = x * x + y * y
    model = camera["model"]
    d = list(distortion) + [0.0] * 8
    if model == "OPENCV_FISHEYE":
        r = np.sqrt(r2)
        theta = np.arctan(r)
        factor = np.divide(
            theta
            * (
                1
                + d[0] * theta**2
                + d[1] * theta**4
                + d[2] * theta**6
                + d[3] * theta**8
            ),
            r,
            out=np.ones_like(r),
            where=r > 0,
        )
        xd, yd = x * factor, y * factor
    else:
        k1, k2 = (d[0], d[1]) if model != "SIMPLE_RADIAL" else (d[0], 0)
        p1, p2 = (d[2], d[3]) if model in ("OPENCV", "FULL_OPENCV") else (0, 0)
        radial = 1 + k1 * r2 + k2 * r2 * r2
        if model == "FULL_OPENCV":
            radial = (radial + d[4] * r2**3) / (
                1 + d[5] * r2 + d[6] * r2**2 + d[7] * r2**3
            )
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = xd * K[0, 0] + K[0, 2]
    v = yd * K[1, 1] + K[1, 2]
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    a = (u - x0)[..., None]
    b = (v - y0)[..., None]
    pixels = np.asarray(image.convert("RGB"), float)

    def sample(ix, iy):
        return pixels[np.clip(iy, 0, height - 1), np.clip(ix, 0, width - 1)]

    result = (
        (1 - a) * (1 - b) * sample(x0, y0)
        + a * (1 - b) * sample(x0 + 1, y0)
        + (1 - a) * b * sample(x0, y0 + 1)
        + a * b * sample(x0 + 1, y0 + 1)
    )
    valid = (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    result[~valid] = 0
    from PIL import Image

    return Image.fromarray(np.uint8(np.clip(result, 0, 255))), K
