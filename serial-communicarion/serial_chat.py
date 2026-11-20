import serial
import platform
import threading

# Detect which device is running this program
if platform.system() == "Windows":
    PORT = "COM5"          # Change this to your USB-TTL COM port
    DEVICE = "Laptop"
else:
    PORT = "/dev/serial0"
    DEVICE = "Raspberry Pi"

BAUD_RATE = 115200

# Open serial port
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)

print(f"{DEVICE} connected using {PORT}")
print("Type a message and press Enter.")
print("--------------------------------")


# Continuously receive messages
def receive_data():
    while True:
        try:
            data = ser.readline().decode(errors="ignore").strip()

            if data:
                print(f"\nReceived: {data}")
                print("Send: ", end="", flush=True)

        except Exception as e:
            print("Receive error:", e)
            break


# Start receiving in background
receiver = threading.Thread(target=receive_data, daemon=True)
receiver.start()


# Continuously send messages
while True:
    try:
        message = input("Send: ")

        if message.lower() == "exit":
            break

        ser.write((message + "\n").encode())

    except KeyboardInterrupt:
        break

ser.close()
print("Serial port closed.")