"""Read explicit fixed image calibration without scene or partition priors."""
from pathlib import Path


def view(frame, packet):
    """Convert existing calibration only; never infer geometry or read numeric arrays."""
    reference = frame.get('original_rgb', frame.get('reference_rgb'))
    if not reference:
        raise ValueError('missing original RGB')
    reference = (packet / reference).resolve()
    if not reference.is_file():
        raise ValueError('missing original RGB file')
    K = frame.get('intrinsics_original', frame.get('K'))
    pose = frame.get('camera_to_world')
    if pose is None and 'rotation' in frame and 'center' in frame:
        pose = [list(row) + [frame['center'][i]] for i, row in enumerate(frame['rotation'])] + [[0, 0, 0, 1]]
    if pose is None and frame.get('camera_to_scene_rotation') is not None and frame.get('scene_center') is not None:
        pose = [list(row) + [frame['scene_center'][i]] for i, row in enumerate(frame['camera_to_scene_rotation'])] + [[0, 0, 0, 1]]
    resolution = frame.get('resolution', [frame.get('width'), frame.get('height')])
    if (not isinstance(K, list) or len(K) != 3 or any(not isinstance(row,list) or len(row) != 3 for row in K)
            or not isinstance(pose,list) or len(pose) != 4 or any(not isinstance(row,list) or len(row) != 4 for row in pose)
            or len(resolution) != 2 or any(not isinstance(n,(int,float)) or n <= 0 for n in resolution)):
        raise ValueError('missing/invalid explicit calibration')
    return dict(id=str(frame.get('index', frame.get('id'))), width=int(resolution[0]), height=int(resolution[1]),
                K=K, camera_to_world=pose, reference_rgb=str(reference))
