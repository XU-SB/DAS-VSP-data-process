# DAS-VSP-data-process

**DAS-VSP-data-process** is a **Python-based** source code package developed by **Dr. Xu Shibo** from **ENEOS Xplora**. It provides an end-to-end framework for processing **Distributed Acoustic Sensing Vertical Seismic Profiling (DAS-VSP)** data, from source wavelet estimation and DAS data conditioning to numerical forward modeling, **Full Waveform Inversion (FWI)**, and **Reverse Time Migration (RTM)** imaging.

The toolkit follows a practical DAS-VSP workflow: estimate the source from geophone VSP data, build subsurface models from sonic logs, process raw DAS traces into clean up-going wavefields, invert for the velocity model using FWI, and generate depth images with RTM around the observation well.

---

## 🔧 Installation

To install **DAS-VSP-data-process**, first clone the repository:

    git clone https://github.com/XU-SB/DAS-VSP-data-process.git
    cd DAS-VSP-data-process

The package is designed to run on **Linux systems**, typically from Python scripts or Jupyter notebooks.

---

## 📦 Required Modules

**Core dependencies**

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`

**Optional (workflow dependent)**

- `lasio` – LAS well log reading  
- `pyseis` – seismic modeling / inversion backend for forward modeling, FWI, and RTM  

Install the required packages using `pip` inside your preferred virtual environment.

---

## 💻 GPU Requirements

FWI and RTM are computationally intensive. A **GPU-capable** backend (e.g., using `pyseis` with CUDA-enabled hardware) is strongly recommended, especially for large models or 2D/3D experiments.

---

## 🧩 Package Overview

The repository is organized into several top-level modules, each corresponding to a key stage in the DAS-VSP workflow:

### `Source/` – Source Wavelet Estimation

- Reads geophone VSP records.
- Computes autocorrelation to estimate the effective source wavelet.
- Outputs a source function that can be reused in forward modeling, FWI, and RTM.

### `ForwardModeling/` – Numerical Forward Modeling

- Builds 1D/2D velocity and density models from sonic logs (and density logs if available).
- Uses the estimated source wavelet to generate synthetic VSP or DAS-VSP data.
- Provides reference synthetic data sets and initial models for testing the processing and inversion flow.

### `DAS-Data-Processing/` – DAS-VSP Processing Workflow

Implements the main DAS-VSP data-processing sequence (corresponding to the “Workflow for Data-Processing_ENEOS” figure):

1. Trim raw DAS data and apply demean.
2. Apply band-limited filtering (e.g., 5–80 Hz).
3. Pick first arrivals using sonic-log-based timing.
4. Estimate the down-going wavefield (corridor-style median or similar approach).
5. Subtract the down-going estimate to obtain the up-going residual.
6. Optionally apply FK separation to further separate up-going and down-going energy.
7. Mute first arrivals and rescale near traces to stabilize amplitudes.

The output is a cleaned up-going DAS wavefield suitable for inversion and imaging.

### `FWI/` – Velocity Inversion from DAS Data

- Takes the processed DAS up-going data and an initial velocity model (from sonic logs or forward modeling).
- Runs FWI to refine the subsurface velocity structure near the well.
- Produces updated velocity models for use in RTM.

A GPU-based modeling engine is assumed for practical run times.

### `RTM/` – Reverse Time Migration Imaging

- Uses the FWI-updated velocity model and processed DAS-VSP data.
- Performs RTM to compute depth-domain reflectivity images around the receiver well.
- Generates migrated images that can be compared with well information and reservoir models.

---

## 🚀 Typical Workflow

A standard end-to-end use of **DAS-VSP-data-process**:

1. Estimate the source wavelet from geophone VSP data in `Source/`.
2. Build velocity/density models from sonic logs and generate synthetic data in `ForwardModeling/`.
3. Process field DAS-VSP data in `DAS-Data-Processing/` (trim, filter, first picks, down-going estimation, subtraction, FK, mute, rescale).
4. Run FWI in `FWI/` to obtain an updated velocity model constrained by the DAS-VSP data.
5. Run RTM in `RTM/` to obtain migrated images using the FWI velocity model.

Each module is independent, so users can replace or extend individual components without modifying the entire workflow.

---

## 📬 Contact

Developer: **Dr. Xu Shibo**, ENEOS Xplora  

For questions or bug reports, please contact:

- `xushibo11@gmail.com`  
- `xu.shibo@eneos.com`
