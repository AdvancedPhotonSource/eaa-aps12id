import numpy as np
import pytest

from eaa_aps12id.tools import SimulatedSpatialSAXS

h5py = pytest.importorskip("h5py")


def write_scan(root, *, scan_prefix="Ssample_00236", spectrum_ids=(1, 2, 3, 4)):
    averaged_dir = root / "SAXS" / "Averaged"
    metadata_dir = root / "Metadata"
    averaged_dir.mkdir(parents=True)
    metadata_dir.mkdir()

    x_by_id = {1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0}
    y_by_id = {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0}
    q = np.array([0.0, 1.0, 2.0, 3.0])
    for spectrum_id in spectrum_ids:
        x = x_by_id[spectrum_id]
        y = y_by_id[spectrum_id]
        intensity = 10.0 * x + 20.0 * y + q
        error = np.full_like(q, 0.1)
        data = np.column_stack([q, intensity, error])
        np.savetxt(
            averaged_dir / f"{scan_prefix}_{spectrum_id:05d}.dat",
            data,
            header="% q I err",
            comments="",
        )

    metadata_name = scan_prefix.rsplit("_", maxsplit=1)[0][1:]
    with h5py.File(metadata_dir / f"{metadata_name}.h5", "w") as h5_file:
        measurement = h5_file.create_group("entry").create_group("measurement")
        measurement.create_dataset("motor_sth", data=[0.0, 1.0, 0.0, 1.0])
        measurement.create_dataset("motor_sav", data=[0.0, 0.0, 1.0, 1.0])

    return q


def test_simulated_spatial_saxs_interpolates_position_and_q_grid(tmp_path):
    write_scan(tmp_path)
    tool = SimulatedSpatialSAXS(tmp_path, "Ssample_00236_*.dat")

    q, intensity = tool.acquire_saxs(x=0.5, y=0.5, q_min=0.0, q_max=3.0, q_step=0.5)

    expected_q = np.arange(0.0, 3.0, 0.5)
    np.testing.assert_allclose(q, expected_q)
    np.testing.assert_allclose(intensity, 15.0 + expected_q)
    assert tool.scan_identifier == "sample_00236"
    assert tool.saxs_data.shape == (2, 2, 4, 2)


def test_spectrum_id_maps_to_one_based_metadata_row(tmp_path):
    write_scan(tmp_path, spectrum_ids=(2,))
    tool = SimulatedSpatialSAXS(tmp_path, "Ssample_00236_00002.dat")

    q, intensity = tool.acquire_saxs(x=1.0, y=0.0, q_min=0.0, q_max=2.0, q_step=1.0)

    np.testing.assert_allclose(q, [0.0, 1.0])
    np.testing.assert_allclose(intensity, [10.0, 11.0])
    np.testing.assert_allclose(tool.x_values, [1.0])
    np.testing.assert_allclose(tool.y_values, [0.0])


def test_missing_dat_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError, match="No SAXS"):
        SimulatedSpatialSAXS(tmp_path, "Ssample_00236_*.dat")


def test_missing_metadata_raises(tmp_path):
    averaged_dir = tmp_path / "SAXS" / "Averaged"
    averaged_dir.mkdir(parents=True)
    np.savetxt(averaged_dir / "Ssample_00236_00001.dat", np.ones((2, 3)))

    with pytest.raises(FileNotFoundError, match="Metadata"):
        SimulatedSpatialSAXS(tmp_path, "Ssample_00236_*.dat")


def test_mixed_scan_identifiers_raise(tmp_path):
    write_scan(tmp_path)
    averaged_dir = tmp_path / "SAXS" / "Averaged"
    np.savetxt(averaged_dir / "Sother_00237_00001.dat", np.ones((2, 3)))

    with pytest.raises(ValueError, match="one scan identifier"):
        SimulatedSpatialSAXS(tmp_path, "S*_*.dat")


def test_zero_spectrum_id_raises(tmp_path):
    write_scan(tmp_path)
    averaged_dir = tmp_path / "SAXS" / "Averaged"
    np.savetxt(averaged_dir / "Ssample_00236_00000.dat", np.ones((2, 3)))

    with pytest.raises(ValueError, match="one-based"):
        SimulatedSpatialSAXS(tmp_path, "Ssample_00236_00000.dat")


def test_incomplete_grid_raises(tmp_path):
    write_scan(tmp_path, spectrum_ids=(1, 2, 3))

    with pytest.raises(ValueError, match="complete rectangular grid"):
        SimulatedSpatialSAXS(tmp_path, "Ssample_00236_*.dat")


def test_out_of_bounds_position_raises(tmp_path):
    write_scan(tmp_path)
    tool = SimulatedSpatialSAXS(tmp_path, "Ssample_00236_*.dat")

    with pytest.raises(ValueError):
        tool.acquire_saxs(x=2.0, y=0.5, q_min=0.0, q_max=2.0, q_step=1.0)


def test_inconsistent_q_grid_raises(tmp_path):
    write_scan(tmp_path)
    path = tmp_path / "SAXS" / "Averaged" / "Ssample_00236_00004.dat"
    data = np.loadtxt(path, comments="%")
    data[:, 0] = [0.0, 1.0, 2.5, 3.0]
    np.savetxt(path, data)

    with pytest.raises(ValueError, match="same q grid"):
        SimulatedSpatialSAXS(tmp_path, "Ssample_00236_*.dat")
