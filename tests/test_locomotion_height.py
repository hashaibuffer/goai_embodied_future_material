import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/s10_terrain_perception"), str(ROOT / "src/s10_terrain_policy")]
from mujoco_teacher import PrivilegedHeightScanner
from locomotion_height import select_support_surface, height_debug_record, wheel_tread_heights
from teacher_skill_router import clamp_height_window


def scene(objects, z=0.):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
        <geom type="plane" size="10 10 .1" pos="0 0 {z}"/>
        {objects}</worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scanner = PrivilegedHeightScanner(model)
    pos = np.array([0., 0., z + .42])
    raw = scanner.scan_geometry(data, pos, np.eye(3))
    result = select_support_surface(scanner, data, pos, np.eye(3), raw)
    return scanner, pos, raw, result


def test_wheel_tread_rays_see_floor_not_overhead_and_reject_missing_support():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="10 10 .1"/>
      <geom type="box" pos=".5 0 .05" size=".5 1 .05"/>
      <geom type="box" pos="0 0 1" size="2 2 .05"/>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scanner = PrivilegedHeightScanner(model)
    wheels = np.array([[.4, .2, .21], [.4, -.2, .21], [-.4, .2, .11], [-.4, -.2, .11]])
    np.testing.assert_allclose(wheel_tread_heights(scanner, data, wheels), [.1, .1, 0., 0.], atol=1e-7)
    wheels[0, 2] = .8
    assert np.isnan(wheel_tread_heights(scanner, data, wheels)[0])


@pytest.mark.parametrize("z", [0., 2.7])
def test_selects_floor_under_slab_not_slab_top_or_underside(z):
    scanner, pos, raw, result = scene(
        f'<geom type="box" size="1 1 .05" pos=".8 0 {z + .8}"/>', z)
    mask = (scanner.local_xy[:, 0] > .2) & (scanner.local_xy[:, 0] < 1.5) & (np.abs(scanner.local_xy[:, 1]) < .8)
    np.testing.assert_allclose(raw.hit_z_w[mask], z + .85)
    np.testing.assert_allclose(result.scan.hit_z_w[mask], z, atol=1e-7)
    np.testing.assert_allclose(result.scan.height[mask], -.08, atol=1e-7)
    assert result.changed[mask].all()
    assert result.conversion_error_m < 1e-10


def test_solid_wall_is_not_replaced_with_floor_beneath_it():
    scanner, pos, raw, result = scene('<geom type="box" size=".3 3 .4" pos=".9 0 .4"/>')
    mask = (scanner.local_xy[:, 0] > .65) & (scanner.local_xy[:, 0] < 1.15)
    np.testing.assert_allclose(result.scan.hit_z_w[mask], .8)
    assert not result.changed[mask].any()


def test_stairs_below_upper_slab_are_preserved():
    objects = ''.join(f'<geom type="box" size=".2 2 {(i+1)*.075/2}" pos="{.6+i*.4} 0 {(i+1)*.075/2}"/>' for i in range(5))
    objects += '<geom type="box" size="1.4 2 .05" pos="1.2 0 1.0"/>'
    scanner, pos, raw, result = scene(objects)
    for i in range(5):
        mask = (np.abs(scanner.local_xy[:, 0] - (.6+i*.4)) < .05) & (np.abs(scanner.local_xy[:, 1]) < .5)
        np.testing.assert_allclose(result.scan.hit_z_w[mask], (i+1)*.075, atol=1e-7)


def test_windowed_rescan_matches_full_selection_where_actor_and_detector_read():
    objects = '<geom type="box" size="1.4 2 .05" pos="1.2 0 1.0"/>'
    scanner, pos, raw, full = scene(objects)
    data = mujoco.MjData(scanner.model)
    mujoco.mj_forward(scanner.model, data)
    limited = select_support_surface(scanner, data, pos, np.eye(3), raw,
                                     query_x_range=[-.4, .8], query_half_width=.25)
    mask = (scanner.local_xy[:, 0] >= -.4) & (scanner.local_xy[:, 0] <= 1.2) & (np.abs(scanner.local_xy[:, 1]) <= .25)
    np.testing.assert_allclose(limited.scan.hit_z_w[mask], full.scan.hit_z_w[mask])
    for value in (limited, full):
        actor = clamp_height_window(value.scan.height.reshape(33, 41).T[None], .25, [-.4, .8])
        np.testing.assert_allclose(actor, -.08, atol=1e-7)


def test_xy_clamp_debug_distinguishes_extension_from_surface():
    scanner, pos, raw, result = scene('<geom type="box" size="1 2 .1" pos="2 0 .1"/>')
    height = result.scan.height.reshape(33, 41).T[None]
    processed = clamp_height_window(height, .25, [-.4, .8])
    debug = height_debug_record(raw, result, processed, pos)
    assert debug["postprocess_changed_count"] > 0
    assert debug["changed_surface_count"] == 0
    assert debug["conversion_error_m"] < 1e-10
    assert debug["raw_conversion_error_m"] < 1e-6
    assert np.allclose(processed, -.08)


def test_xy_clamp_keeps_real_inner_values_and_each_axis_boundary():
    x, y = np.linspace(-.8, 3.2, 41), np.linspace(-1.6, 1.6, 33)
    height = (2*x[:, None] + 3*y[None])[None].astype(np.float32)
    before = height.copy()
    actual = clamp_height_window(height, .25, [-.35, 1.15])
    expected = 2*np.clip(x, -.35, 1.15)[:, None] + 3*np.clip(y, -.25, .25)[None]
    np.testing.assert_allclose(actual[0], expected, atol=1e-6)
    np.testing.assert_array_equal(height, before)
    with pytest.raises(ValueError):
        clamp_height_window(height, .25, [1., 0.])


def test_height_markers_are_disabled_even_with_legacy_cli_controls():
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT / "scripts"))
    from play_mujoco_teacher import draw_height_layers, draw_actor_height_debug, build_arg_parser
    scanner, pos, raw, result = scene('<geom type="box" size="1 1 .05" pos=".8 0 .8"/>')
    viewer = SimpleNamespace(user_scn=mujoco.MjvScene(scanner.model, maxgeom=5000))
    draw_height_layers(viewer, raw, result, result.scan.height.reshape(33, 41).T[None], pos, "all")
    draw_actor_height_debug(viewer, None, None, None)
    assert viewer.user_scn.ngeom == 0
    args = build_arg_parser().parse_args(["--xml", "map.xml", "--low-support-surface",
                                         "--low-height-x-range", "-.4", "1.2",
                                         "--height-debug-layer", "all"])
    assert args.low_support_surface and args.low_height_x_range == [-.4, 1.2]
    assert args.viewer_hz == 30 and args.actor_threads == 1
