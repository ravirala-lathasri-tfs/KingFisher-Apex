import serial

PORT = "/dev/serial0"
BAUD_RATE = 115200

ser = serial.Serial(PORT, BAUD_RATE, timeout=1)

print("Raspberry Pi Serial Sender")
print("--------------------------")
print("Examples:")
print("text:HELLO")
print("hex:AA 01 FF 55")
print("bin:10101010 00000001 11111111 01010101")
print("Type 'exit' to stop.")
print()


while True:
    message = input("Send: ").strip()

    if message.lower() == "exit":
        break

    try:
        # TEXT INPUT
        if message.lower().startswith("text:"):
            text_data = message[5:]
            data = text_data.encode("utf-8")

        # HEX INPUT
        elif message.lower().startswith("hex:"):
            hex_data = message[4:]
            data = bytes.fromhex(hex_data)

        # BINARY INPUT
        elif message.lower().startswith("bin:"):
            binary_data = message[4:].split()

            data = bytes(
                int(byte, 2)
                for byte in binary_data
            )

        else:
            print("Invalid format.")
            print("Use text:, hex:, or bin:")
            continue

        # Send raw bytes
        ser.write(data)

        print("Data sent successfully.")

    except ValueError:
        print("Invalid HEX or binary data.")


ser.close()

print("Serial port closed.")