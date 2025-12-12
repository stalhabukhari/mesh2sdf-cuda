import math
import argparse
from pathlib import Path

import numpy as np
import torch
from lib.datasets import MeshSDFDatasetSimple
from lib.torchgp import load_obj


def sample_from_objfile(
    obj_filepath: Path,
    num_samples_surf: int,
    num_samples_sdf: int,
    chunk_size: int,
    seed: int,
):
    sample_mode = ["trace", "near", "rand"]
    samples_per_tech = [
        num_samples_surf,
        math.ceil(num_samples_sdf * 0.7),
        math.ceil(num_samples_sdf * 0.3),
    ]
    samples_done = [0, 0, 0]

    samples_dict = {
        "trace": (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
        ),
        "near": (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
        ),
        "rand": (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
        ),
    }

    done = False
    while not done:
        samples_per_tech_now = [
            max(tot - d, 0) for tot, d in zip(samples_per_tech, samples_done)
        ]
        if sum(samples_per_tech_now) == 0:
            done = True
            break

        samples_per_tech_now = [min(tot, chunk_size) for tot in samples_per_tech_now]

        meshds = MeshSDFDatasetSimple(
            dataset_path=obj_filepath,
            sample_mode=sample_mode,
            get_normals=False,
            seed=seed,
            samples_per_tech=samples_per_tech_now,
            noisy_trace_var=None,
        )

        pts = meshds.pts.numpy()
        sdf = meshds.d.numpy()

        pts_trace = pts[: samples_per_tech_now[0]]
        pts_near = pts[
            samples_per_tech_now[0] : samples_per_tech_now[0] + samples_per_tech_now[1]
        ]
        pts_rand = pts[samples_per_tech_now[0] + samples_per_tech_now[1] :]
        sdf_trace = sdf[: samples_per_tech_now[0]]
        sdf_near = sdf[
            samples_per_tech_now[0] : samples_per_tech_now[0] + samples_per_tech_now[1]
        ]
        sdf_rand = sdf[samples_per_tech_now[0] + samples_per_tech_now[1] :]

        pts_trace = np.concatenate([samples_dict["trace"][0], pts_trace], axis=0)
        pts_near = np.concatenate([samples_dict["near"][0], pts_near], axis=0)
        pts_rand = np.concatenate([samples_dict["rand"][0], pts_rand], axis=0)
        pts_trace_sdf = np.concatenate([samples_dict["trace"][1], sdf_trace], axis=0)
        pts_near_sdf = np.concatenate([samples_dict["near"][1], sdf_near], axis=0)
        pts_rand_sdf = np.concatenate([samples_dict["rand"][1], sdf_rand], axis=0)
        samples_dict["trace"] = (pts_trace, pts_trace_sdf)
        samples_dict["near"] = (pts_near, pts_near_sdf)
        samples_dict["rand"] = (pts_rand, pts_rand_sdf)

        samples_done = [
            samples_done[i] + samples_per_tech_now[i] for i in range(len(samples_done))
        ]

    trace_samps = samples_per_tech[0]
    near_samps = samples_per_tech[1]
    rand_samps = samples_per_tech[2]
    samples_dict["trace"] = (
        samples_dict["trace"][0][:trace_samps],
        samples_dict["trace"][1][:trace_samps],
    )
    samples_dict["near"] = (
        samples_dict["near"][0][:near_samps],
        samples_dict["near"][1][:near_samps],
    )
    samples_dict["rand"] = (
        samples_dict["rand"][0][:rand_samps],
        samples_dict["rand"][1][:rand_samps],
    )

    assert len(samples_dict["trace"][0]) == trace_samps
    assert len(samples_dict["near"][0]) == near_samps
    assert len(samples_dict["rand"][0]) == rand_samps

    assert len(samples_dict["trace"][1]) == trace_samps
    assert len(samples_dict["near"][1]) == near_samps
    assert len(samples_dict["rand"][1]) == rand_samps

    return samples_dict


def norm_params_from_objfile(obj_filepath: Path):
    """
    Normalized mesh params as computed in MeshSDFDatasetSimple
    """
    assert obj_filepath.exists(), f"{obj_filepath} does not exist."
    out = load_obj(str(obj_filepath))
    V, F = out
    # Normalize mesh
    V_max, _ = torch.max(V, dim=0)
    V_min, _ = torch.min(V, dim=0)
    V_center = (V_max + V_min) / 2.0
    V = V - V_center

    # Find the max distance to origin
    max_dist = torch.sqrt(torch.max(torch.sum(V**2, dim=-1)))
    V_scale = 1.0 / max_dist
    V *= V_scale

    return (
        V_center.cpu().numpy(),
        V_scale.cpu().numpy(),
        V.cpu().numpy(),
        F.cpu().numpy(),
    )


def process_dataset(
    dataset_dir: Path,
    save_dir: Path,
    num_samples_surf: int,
    num_samples_sdf: int,
    chunk_size: int,
    seed: int,
):
    """
    Process meshes in a directory
    """
    assert dataset_dir.exists(), f"{dataset_dir} does not exist."
    save_dir.mkdir(exist_ok=True)

    for obj_cat in dataset_dir.glob("*"):
        assert obj_cat.is_dir(), f"{obj_cat} is not a directory."
        cat_name = obj_cat.name
        for obj_file in obj_cat.glob("*.obj"):
            obj_name = obj_file.stem

            print(f" - Processing: {obj_file}")
            samples_dict = sample_from_objfile(
                str(obj_file), num_samples_surf, num_samples_sdf, chunk_size, seed
            )

            save_path = save_dir / cat_name
            save_path.mkdir(exist_ok=True)

            np.save(save_path / f"{obj_name}-trace-pts.npy", samples_dict["trace"][0])
            np.save(save_path / f"{obj_name}-near-pts.npy", samples_dict["near"][0])
            np.save(save_path / f"{obj_name}-rand-pts.npy", samples_dict["rand"][0])
            np.save(save_path / f"{obj_name}-trace-sdf.npy", samples_dict["trace"][1])
            np.save(save_path / f"{obj_name}-near-sdf.npy", samples_dict["near"][1])
            np.save(save_path / f"{obj_name}-rand-sdf.npy", samples_dict["rand"][1])

            v_center, v_scale, mesh_v, mesh_f = norm_params_from_objfile(obj_file)
            np.save(save_path / f"{obj_name}-mesh_v.npy", mesh_v)
            np.save(save_path / f"{obj_name}-mesh_f.npy", mesh_f)
            np.save(save_path / f"{obj_name}-v_center.npy", v_center)
            np.save(save_path / f"{obj_name}-v_scale.npy", v_scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sample points an SDF from mesh")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--num_samples_surf", type=int, default=100_000)
    parser.add_argument("--num_samples_sdf", type=int, default=600_000)
    parser.add_argument("--chunk_size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    print(f"Configuration:\n{args}")

    process_dataset(
        args.dataset_dir,
        args.save_dir,
        args.num_samples_surf,
        args.num_samples_sdf,
        args.chunk_size,
        args.seed,
    )
