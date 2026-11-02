# PYNQ Notebooks and Runtime Code

This folder contains Jupyter notebooks and supporting Python scripts for sensor testing, data collection, and live ankle assistance on the PYNQ-Z2.

| File | Purpose |
|---|---|
| `logger.ipynb` | Notebook for recording sensor data. |
| `LiveWalking.ipynb` | Notebook for live intent prediction and ankle assistance. |
| `sensor_check (1).py` | Sensor connection and reading checks. |
| `safety_layer (1).py` | Safety checks and assistance gating. |
| `run_afo.py` | Runtime code for operating the ankle exoskeleton. |
| `pynq_data_logger (1).py` | Supporting sensor data logging code. |
| `can_servo (1).py` | Motor command and communication helpers. |
| `can_mcp.py` | Interface code for the Pmod CAN controller. |

## Usage

Copy the required notebooks, supporting scripts, and matching `.bit` and `.hwh` overlay files to the PYNQ-Z2. Update local paths as needed and run notebook cells in order.
