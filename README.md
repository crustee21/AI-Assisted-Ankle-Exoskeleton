# AI-Assisted-Ankle-Exoskeleton
This ankle exoskeleton aids individuals with foot drop by providing motor assistance when the wearer intends to walk.


# Motivation
I have foot drop caused by a condition called distal spinal muscular atrophy, and I have been wearing traditional ankle-foot orthoses (AFOs) for almost 7 years. I faced many issues with them snapping or just being uncomfortable in general. They hold the ankle in a fixed position, promoting muscle atrophy from disuse and make it difficult to climb uphill/stairs. I dreamed of a device that could still let your ankle move freely but also provide assistance when needed. Prior to building this ambitious project, I had very little experience with machine learning, FPGA programming, and electronics in general. Though it is far from perfect, I am proud of how far this project has come over the years and am excited to improve it the more experienced I become.

# How it works

To predict walk intent, the device incorporates 3 types of sensors: an Inertial Measurement Unit (IMU), electromyography sensor (EMG), and Force Sensitive Resistors (FSRs). The sensor data is fed into a convolutional neural network with long short-term memory flashed onto a field programmable gate array (FPGA). A deterministic safety layer checks the FSRs on the bottom of the footplate to ensure the physical evidence agrees with the algorithm's guess. The wearer's foot goes in the footplate which is attached to the motor and the strut. The model successfully predicts walk intent an average of 360 ms before toe-off.

# Current status

The model successfully performed real-time intent prediction and provided powered ankle assitance.

### Completed
- Designed and 3D printed the footplate and motor mounting structure around my own leg
- Collected personal EMG and IMU walking data
- Trained a CNN-LSTM with data from ENABL3S and personal data
- Integrated live intent prediction with motor control on the FPGA
- Demonstrated powered ankle assistance using battery-powered system

### Future Improvements
- Refining the 3D printed components to accommodate a wider range of users
- Designing PCB for better wire organization
- Improving structural strength and integrity
- Improving comfort, fit, and ease of use
- Organizing and securing wiring for more reliable use

### Further Evaluation
- Consistency of assistance across repeated walking trials
- Long-term mechanical durability
- Effects on gait and muscular effort

# Hardware
### Processing and communication
- **PYNQ-Z2 FPGA:** Runs sensor acquisition, signal processing, live intent prediction, and motor control
- **Diligent Pmod CAN:** Provides CAN communication between the PYNQ-Z2 and motor
- **ADS1015 12-Bit ADC:** Digitizes the EMG's analog signal
- **ISO1540 I2C isolator:** Isolates communication with the separately powered EMG
### Sensors
- **MyoWare 2.0:** Measures muscle activity over the tibialis anterior
- **BMI160 IMU:** Mounted over the anterior tibia, measures acceleration and angular velocity
- **Two FSR406B force-sensitive resistors:** Detect contact at the heel and big toe for stance-based safety gating
### Actuation and Structure
- **CubeMars AK70-10 KV100 motor:** Provides powered ankle assistance
- **Custom 3D-printed footplate and mounting components:** Support the foot and secure the actuator, printed with PLA+ filament
- **Straps and padding:** Secure the device and sensors to the leg.
- **Cable sleeves & Fanny Pack:** Organize wiring
### Power
- **Two 4s LiPo batteries:** When connected in series, supply the motor and FPGA
- **Pololu 5 V step-down regulator D36V28F5:** Powers the PYNQ-Z2
- **3.7 V 100 mAh LiPo:** Powers the EMG electronics
- **TP4056:** USB-C charging port for EMG's battery
- **Other:** ATO fuse holder (15 amps), XT60 Series Splitter, XT90 anti-spark connectors, NC emergency stop, Low voltage buzzer alarms
### Software
- **AMD Vivado & Vitis:** Builds the FPGA design, including sensor and communication interfaces
- **PYNQ and Python:** Provide hardware access and run the data collection and live assistance pipeline
- **PyTorch:** Machine Learning framework
- **Jupyter notebooks:** Support data collection, model development, and live testing

The project uses a field programmable gate array (FPGA) with a convolutional neural network with long short-term memory (CNN-LSTM) that predicts walk intent. The CNN-LSTM uses an inertial measurement unit (IMU) and an electromyography sensor (EMG) to make its prediction. There is also a deterministic safety layer that checks two force sensitive resistors (FSRs) on the bottom of the shoe to ensure that the physical evidence agrees with the model's prediction.
