# I2C (for BMI160 and ADS1015) on Arduino heaer SCL/SDA pins
set_property -dict {PACKAGE_PIN P15 IOSTANDARD LVCMOS33} [get_ports IIC_0_scl_io]
set_property -dict {PACKAGE_PIN P16 IOSTANDARD LVCMOS33} [get_ports IIC_0_sda_io]

# XADC for FSRs (heel is A0 on vaux1 and toe is A1 on vaux9)
set_property -dict {PACKAGE_PIN E17 IOSTANDARD LVCMOS33} [get_ports Vaux1_0_v_p]   ;# ar_an0_p
set_property -dict {PACKAGE_PIN D18 IOSTANDARD LVCMOS33} [get_ports Vaux1_0_v_n]   ;# ar_an0_n
set_property -dict {PACKAGE_PIN E18 IOSTANDARD LVCMOS33} [get_ports Vaux9_0_v_p]   ;# ar_an1_p
set_property -dict {PACKAGE_PIN E19 IOSTANDARD LVCMOS33} [get_ports Vaux9_0_v_n]   ;# ar_an1_n

# Pmod CAN for PMODA
# J1: 1 ~CS  2 MOSI  3 MISO  4 SCK  7 INT  8 ~RST.   axi_quad_spi: io0 = MOSI, io1 = MISO
set_property -dict {PACKAGE_PIN Y18 IOSTANDARD LVCMOS33} [get_ports SPI_0_0_ss_io]    ;# JA1
set_property -dict {PACKAGE_PIN Y19 IOSTANDARD LVCMOS33} [get_ports SPI_0_0_io0_io]   ;# JA2
set_property -dict {PACKAGE_PIN Y16 IOSTANDARD LVCMOS33} [get_ports SPI_0_0_io1_io]   ;# JA3
set_property -dict {PACKAGE_PIN Y17 IOSTANDARD LVCMOS33} [get_ports SPI_0_0_sck_io]   ;# JA4
set_property -dict {PACKAGE_PIN U18 IOSTANDARD LVCMOS33} [get_ports {can_int[0]}]     ;# JA7
set_property -dict {PACKAGE_PIN U19 IOSTANDARD LVCMOS33} [get_ports {can_rst_n[0]}]   ;# JA8
set_property PULLUP true [get_ports SPI_0_0_io1_io]

## slow asynchronous I/O
set_false_path -from [get_ports {SPI_0_0_io1_io {can_int[0]} IIC_0_scl_io IIC_0_sda_io}]
set_false_path -to   [get_ports {SPI_0_0_io0_io SPI_0_0_sck_io SPI_0_0_ss_io {can_rst_n[0]} IIC_0_scl_io IIC_0_sda_io}]
