# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# import os

import torch
from torch.utils.data import Dataset

from lib.torchgp import (
    load_obj,
    point_sample,
    point_sample_per_tech,
    sample_surface,
    compute_sdf,
    normalize,
)

from lib.utils import PerfTimer, setparam


class MeshSDFDataset(Dataset):
    """Base class for single mesh datasets."""

    def __init__(
        self,
        args=None,
        dataset_path=None,
        raw_obj_path=None,
        sample_mode=None,
        get_normals=None,
        seed=None,
        num_samples=None,
        trim=None,
        sample_tex=None,
    ):
        self.args = args
        self.dataset_path = setparam(args, dataset_path, "dataset_path")
        self.raw_obj_path = setparam(args, raw_obj_path, "raw_obj_path")
        self.sample_mode = setparam(args, sample_mode, "sample_mode")
        self.get_normals = setparam(args, get_normals, "get_normals")
        self.num_samples = setparam(args, num_samples, "num_samples")
        self.trim = setparam(args, trim, "trim")
        self.sample_tex = setparam(args, sample_tex, "sample_tex")

        # Possibly remove... or fix trim obj
        # if self.raw_obj_path is not None and not os.path.exists(self.dataset_path):
        #    _, _, self.mesh = trim_obj_to_file(self.raw_obj_path, self.dataset_path)
        # elif not os.path.exists(self.dataset_path):
        #    assert False and "Data does not exist and raw obj file not specified"
        # else:

        if self.sample_tex:
            out = load_obj(self.dataset_path, load_materials=True)
            self.V, self.F, self.texv, self.texf, self.mats = out
        else:
            self.V, self.F = load_obj(self.dataset_path)

        self.V, self.F = normalize(self.V, self.F)
        self.mesh = self.V[self.F]
        self.resample()

    def resample(self):
        """Resample SDF samples."""

        self.nrm = None
        if self.get_normals:
            self.pts, self.nrm = sample_surface(self.V, self.F, self.num_samples * 5)
            self.nrm = self.nrm.cpu()
        else:
            self.pts = point_sample(self.V, self.F, self.sample_mode, self.num_samples)
            if "trace" in self.sample_mode:
                print(f"Adding gaussian noise (std=0.1) to trace samples")
                idx = self.sample_mode.index("trace")
                n = self.num_samples // 2
                self.pts[idx * n : (idx + 1) * n] += (
                    torch.randn_like(self.pts[idx * n : (idx + 1) * n]) * 0.1
                )

        self.d = compute_sdf(self.V.cuda(), self.F.cuda(), self.pts.cuda())

        self.d = self.d[..., None]
        self.d = self.d.cpu()
        self.pts = self.pts.cpu()

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.get_normals:
            return self.pts[idx], self.d[idx], self.nrm[idx]
        elif self.sample_tex:
            return self.pts[idx], self.d[idx], self.rgb[idx]
        else:
            return self.pts[idx], self.d[idx]

    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.pts.size()[0]

    def num_shapes(self):
        """Return length of dataset (number of _mesh models_)."""

        return 1


class MeshSDFDatasetSimple(MeshSDFDataset):
    def __init__(
        self,
        args=None,
        dataset_path=None,
        raw_obj_path=None,
        sample_mode=None,
        get_normals=None,
        seed=None,
        samples_per_tech=None,
        trim=None,
        sample_tex=None,
        noisy_trace_var=None,  # usually 0.1
    ):
        self.args = args
        self.dataset_path = setparam(args, dataset_path, "dataset_path")
        self.raw_obj_path = setparam(args, raw_obj_path, "raw_obj_path")
        self.sample_mode = setparam(args, sample_mode, "sample_mode")
        self.get_normals = setparam(args, get_normals, "get_normals")
        self.samples_per_tech = setparam(args, samples_per_tech, "samples_per_tech")
        self.trim = setparam(args, trim, "trim")
        self.sample_tex = setparam(args, sample_tex, "sample_tex")
        self.noisy_trace_var = setparam(args, noisy_trace_var, "noisy_trace_var")

        # Possibly remove... or fix trim obj
        # if self.raw_obj_path is not None and not os.path.exists(self.dataset_path):
        #    _, _, self.mesh = trim_obj_to_file(self.raw_obj_path, self.dataset_path)
        # elif not os.path.exists(self.dataset_path):
        #    assert False and "Data does not exist and raw obj file not specified"
        # else:

        if self.sample_tex:
            out = load_obj(self.dataset_path, load_materials=True)
            self.V, self.F, self.texv, self.texf, self.mats = out
        else:
            self.V, self.F = load_obj(self.dataset_path)

        self.V, self.F = normalize(self.V, self.F)
        self.mesh = self.V[self.F]
        self.resample()

    def resample(self):
        """Resample SDF samples."""

        self.nrm = None
        assert not self.get_normals, "Normals not supported in MeshSDFDatasetSimple"
        # self.pts = point_sample(self.V, self.F, self.sample_mode, self.num_samples)
        self.pts = point_sample_per_tech(
            self.V, self.F, self.sample_mode, self.samples_per_tech
        )
        if (
            "trace" in self.sample_mode and self.noisy_trace_var is not None
        ):  # simulating noisy trace
            print(f"Adding gaussian noise (std=0.1) to trace samples")
            idx = self.sample_mode.index("trace")
            num_samples = self.samples_per_tech[idx]
            start = sum(self.samples_per_tech[: max(0, idx - 1)])
            self.pts[start : start + num_samples] += (
                torch.randn_like(self.pts[start : start + num_samples])
                * self.noisy_trace_var
            )

        self.d = compute_sdf(self.V.cuda(), self.F.cuda(), self.pts.cuda())

        self.d = self.d[..., None]
        self.d = self.d.cpu()
        self.pts = self.pts.cpu()
