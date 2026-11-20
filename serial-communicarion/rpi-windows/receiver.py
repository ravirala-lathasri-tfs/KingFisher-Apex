import serial

PORT = "COM5"       # Change if your USB-TTL uses another COM port
BAUD_RATE = 115200

ser = serial.Serial(PORT, BAUD_RATE, timeout=1)

print("Windows HEX Receiver")
print("--------------------")
print("Waiting for data...")


while True:
    try:
        if ser.in_waiting > 0:

            # Receive raw bytes
            data = ser.read(ser.in_waiting)

            # Display ONLY hexadecimal
            print(data.hex(" ").upper())

    except KeyboardInterrupt:
        break


ser.close()

print("Serial port closed.")