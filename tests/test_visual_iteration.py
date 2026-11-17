"""Exercise the RGB-only previews and readable difference cards."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
import iteration as preview
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_rgb_preview_does_not_load_depth_or_evaluate_masks(tmp_path,monkeypatch):
    from PIL import Image
    Image.new('RGB',(64,32)).save(tmp_path/'rgb.png')
    protocol=tmp_path/'protocol.json'
    write(protocol,{'frames':[{'id':'view','reference':'does-not-exist.npz','reference_rgb':'rgb.png',
        'appearance_mask':'missing.npy','appearance_calibration':{'reference':{'K':[[20,0,32],[0,20,16],[0,0,1]],
        'camera_to_world':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}}}]})
    monkeypatch.setattr(preview.np,'load',lambda *a,**kw:pytest.fail('RGB preview opened numeric evidence'))
    assert preview.preview_frames(protocol)[0]['width']==64


def test_difference_marks_only_changed_pixels_with_fixed_color_scale():
    import numpy as np
    from PIL import Image
    before=np.zeros((8,12,3),dtype=np.uint8)
    after=before.copy();after[2:4,3:7,0]=64;after[0,0]=2
    image,count=preview.difference_image(Image.fromarray(before),Image.fromarray(after))
    values=np.asarray(image)
    assert count==8
    assert not values[0,0].any() and not values[7,11].any()
    assert tuple(values[2,3])==(255,230,60)
    identical,count=preview.difference_image(Image.fromarray(before),Image.fromarray(before))
    assert count==0 and not np.asarray(identical).any()
