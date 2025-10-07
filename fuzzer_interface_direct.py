import os
import time
import subprocess
import serial
import struct

# ---- Configurations ----
STATE_FILE = "/tmp/stm_flash_state"
FEATURE_FIFO = "/home/jfeng/Desktop/Research/Debugging/tee_seer/stm_trust/j/fuzzing_paper/fuzzer/bash_script/feature_fifo"
PORT = "/dev/ttyACM0"
CHUNK_SIZE = 2
DELAY = 0.001

ZEPHYR_PROJECT_DIR = "/home/jfeng/zephyrproject_j"
ZEPHYR_SDK_DIR = "/home/jfeng/zephyr-sdk-0.15.1"
VENV_ACTIVATE = os.path.expanduser("~/zephyrproject_j/.venv/bin/activate")
WEST_BUILD_DIR = "/home/jfeng/Desktop/Research/Debugging/tee_seer/data/test/stm_trust/j/fuzzing_paper/syringe_final"

SYRINGE_PATH = "/home/jfeng/Desktop/Research/Debugging/tee_seer/stm_trust/j/fuzzing_paper/fuzzer/bash_script/syringe.bin"  # Modify this if needed


# ---- Flashing ----
def flash_board_if_needed():
    if os.path.isfile(STATE_FILE):
        return  # Already flashed
    print("[*] Flashing STM board...")
    flash_cmd = f"""
        source {VENV_ACTIVATE} && \
        export ZEPHYR_SDK_INSTALL_DIR={ZEPHYR_SDK_DIR} && \
        cd {ZEPHYR_PROJECT_DIR} && \
        west flash --build-dir {WEST_BUILD_DIR}
    """
    result = subprocess.run(flash_cmd, shell=True, executable="/bin/bash",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("[!] Flash failed:")
        print(result.stderr)
        raise RuntimeError("Board flashing failed.")
    print(result.stdout)
    time.sleep(10)
    with open(STATE_FILE, "w") as f:
        f.write("flashed")


# ---- Serial send (bytes only) ----
def send_bytes_over_serial(serial_port, data_bytes, chunk_size=CHUNK_SIZE, delay=DELAY):
    filesize = len(data_bytes)
    offset = 0
    print(f"[*] Sending {filesize} bytes to device...")

    mv = memoryview(data_bytes)  # avoids extra copy
    while offset < filesize:
        end = min(offset + chunk_size, filesize)
        chunk = mv[offset:end]
        serial_port.write(chunk)
        serial_port.flush()
        time.sleep(delay)
        offset = end

    print("[*] Done sending bytes.")


# ---- FIFO Listening ----
def listen_to_fifos(timeout=5.0):
    chunk_ids = []
    if not os.path.exists(FEATURE_FIFO):
        raise FileNotFoundError("Missing FEATURE FIFO.")

    start = time.time()
    while True:
        try:
            with open(FEATURE_FIFO, "rb") as feature_fifo:
                raw = feature_fifo.read(8)
                if len(raw) == 8:
                    crash_flag, num_samples = struct.unpack("<ii", raw)
                    print(f"[*] Received feature vector: crash={crash_flag}, samples={num_samples}")
                    return crash_flag, num_samples, chunk_ids
                else:
                    print("[!] Incomplete feature vector received.")
                    return 0, 0, chunk_ids
        except Exception:
            if time.time() - start > timeout:
                print("[!] FIFO listen timed out.")
                return 0, 0, chunk_ids
            time.sleep(0.05)


# ---- High-level function: accepts ONLY bytes ----
def run_fuzz_cycle_bytes(data_bytes, flash_if_needed=True):
    if not isinstance(data_bytes, bytes):
        raise TypeError("run_fuzz_cycle_bytes expects a bytes object")

    if flash_if_needed:
        flash_board_if_needed()

    if not os.path.exists(PORT):
        raise FileNotFoundError(f"Serial port not found: {PORT}")

    with serial.Serial(PORT, baudrate=115200, timeout=1) as ser:
        send_bytes_over_serial(ser, data_bytes)

    return listen_to_fifos()


'''
# ---- Main Entry Point ----
def main():
    if not os.path.exists(SYRINGE_PATH):
        print(f"[!] syringe.bin not found at path: {SYRINGE_PATH}")
        return

    with open(SYRINGE_PATH, "rb") as f:
        syringe_data = f.read()

    print(f"[*] Read {len(syringe_data)} bytes from {SYRINGE_PATH}")

    crash_flag, num_samples, chunk_ids = run_fuzz_cycle_bytes(syringe_data)

    print("\n[+] Fuzz Cycle Result:")
    print(f"    Crash Flag   : {crash_flag}")
    print(f"    Num Samples  : {num_samples}")
    print(f"    Chunk IDs    : {chunk_ids}")


if __name__ == "__main__":
    main()
'''