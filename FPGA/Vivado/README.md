# Vivado FPGA Design

This folder contains the files needed to recreate the ankle exoskeleton’s Vivado project and load its hardware design onto the PYNQ-Z2.

The HLS model implementation and its verification files are documented separately in [`../hls/`](../hls/).

## Requirements

- AMD Vivado 2025.1
- PYNQ-Z2 board definition files
- A PYNQ-Z2 running a compatible PYNQ image

Vivado is required to rebuild or modify the design. It is not required to load the provided bitstream through PYNQ.

## Folder Contents

| File or folder | Description |
|---|---|
| `board_files/` | PYNQ-Z2 board definitions used to configure the project. |
| `footdrop.tcl` | Tcl script that recreates the Vivado project and block design. |
| `constraints/footdrop.xdc` | Pin assignments and electrical constraints for external connections. |
| `overlay/footdrop.bit` | Compiled FPGA bitstream. |
| `overlay/footdrop.hwh` | Hardware metadata used by PYNQ to identify IP blocks, registers, and connections. |

## Board Files

Install the PYNQ-Z2 board definitions before running the recreation script. Follow the [PYNQ board setup instructions](https://pynq.readthedocs.io/en/v2.5.1/overlay_design_methodology/board_settings.html).

These files describe the board and its configuration. The project-specific external pin assignments are defined separately in the XDC file.

## Recreating the Vivado Project

In Vivado’s Tcl Console, change to this folder and run:

```tcl
cd {C:/path/to/repository/vivado}
source footdrop.tcl
```

Validate the block design and generate the bitstream (run implementation and synthesis first)

The footdrop.tcl script restores the project setup; generating the bitstream is a separate build step.

## Constraints

`constraints/footdrop.xdc` defines the FPGA pin assignments and electrical settings for the external interfaces.

Check these assignments against your physical wiring before connecting hardware. Changes to the pin assignments require rebuilding the bitstream.

## Loading the Provided Overlay

Copy both files from `overlay/` into the same directory on the PYNQ-Z2:

```text
footdrop.bit
footdrop.hwh
```

They must have the same base name and correspond to the same hardware build.

From a Python notebook on the board (Use Jupyter Notebooks), run:

```python
from pynq import Overlay

overlay = Overlay("/home/xilinx/path/to/footdrop.bit")
```

To inspect the detected IP blocks:

```python
print(overlay.ip_dict.keys())
```

Loading the overlay configures the FPGA hardware. Run the project’s separate Python software to acquire sensor data and operate the system.

## Updating the Overlay

After changing the Vivado design, regenerate the bitstream and replace both the `.bit` and `.hwh` files with the corresponding updated versions.

Load the overlay before initializing hardware drivers.
