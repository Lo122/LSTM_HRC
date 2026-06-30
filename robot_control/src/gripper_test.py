
import socket
import struct
import time
import binascii
 
# ---------------------------------------------------------------------------
# Connection settings
# ---------------------------------------------------------------------------
ROBOT_IP   = "169.254.130.206"
MODBUS_PORT = 502        # standard Modbus TCP port
UNIT_ID     = 0x09       # Robotiq gripper Slave ID
TIMEOUT     = 3.0        # socket receive timeout (seconds)
 
# ---------------------------------------------------------------------------
# Modbus TCP framing
# ---------------------------------------------------------------------------
_transaction_id = 0
 
def _next_tid() -> int:
    global _transaction_id
    _transaction_id = (_transaction_id + 1) & 0xFFFF
    return _transaction_id
 
 
def build_tcp_fc16(start_reg: int, data_bytes: bytes) -> bytes:
    """
    FC16 – Preset Multiple Registers (write gripper output registers).
 
    :param start_reg:  Starting register address (e.g. 0x03E8)
    :param data_bytes: Register payload – must be an even number of bytes
    :returns: Complete Modbus TCP frame (MBAP header + PDU, no CRC)
    """
    assert len(data_bytes) % 2 == 0
    num_regs = len(data_bytes) // 2
    pdu = struct.pack(">BHHB", 0x10, start_reg, num_regs, len(data_bytes)) + data_bytes
    # MBAP: Transaction ID (2) + Protocol ID (2) + Length (2) + Unit ID (1)
    mbap = struct.pack(">HHHB", _next_tid(), 0x0000, 1 + len(pdu), UNIT_ID)
    return mbap + pdu
 
 
def build_tcp_fc03(start_reg: int, num_regs: int) -> bytes:
    """
    FC03 – Read Holding Registers (read gripper input registers).
 
    :param start_reg: Starting register address (e.g. 0x07D0)
    :param num_regs:  Number of 16-bit registers to read
    :returns: Complete Modbus TCP frame
    """
    pdu = struct.pack(">BHH", 0x03, start_reg, num_regs)
    mbap = struct.pack(">HHHB", _next_tid(), 0x0000, 1 + len(pdu), UNIT_ID)
    return mbap + pdu
 
 
# ---------------------------------------------------------------------------
# Pre-built command frames
# ---------------------------------------------------------------------------
 
# Activation step 1 – clear rACT
CMD_CLEAR_ACTIVATE = build_tcp_fc16(
    0x03E8,
    bytes([0x00, 0x00,   # rACT=0 (deactivate / reset)
           0x00, 0x00,   # position=0
           0x00, 0x00])  # speed=0, force=0
)
 
# Activation step 2 – set rACT (must remain set after activation)
CMD_ACTIVATE = build_tcp_fc16(
    0x03E8,
    bytes([0x01, 0x00,   # rACT=1 (activate)
           0x00, 0x00,
           0x00, 0x00])
)
 
# Read 1 status register (enough to check gSTA during activation)
CMD_READ_STATUS_1REG = build_tcp_fc03(0x07D0, 1)
 
# Read 3 status registers (full status: position, current, fault)
CMD_READ_STATUS_3REG = build_tcp_fc03(0x07D0, 3)
 
# Close – rACT=1, rGTO=1, rPR=0xFF (fully closed), rSP=0xFF, rFR=0xFF
CMD_CLOSE = build_tcp_fc16(
    0x03E8,
    bytes([0x09, 0x00,   # rACT=1, rGTO=1
           0x00, 0xFF,   # rPR=0xFF (fully closed)
           0xFF, 0xFF])  # rSP=max, rFR=max
)
 
# Open – rACT=1, rGTO=1, rPR=0x00 (fully open), rSP=0xFF, rFR=0xFF
CMD_OPEN = build_tcp_fc16(
    0x03E8,
    bytes([0x09, 0x00,   # rACT=1, rGTO=1
           0x00, 0x00,   # rPR=0x00 (fully open)
           0xFF, 0xFF])  # rSP=max, rFR=max
)
 
 
# ---------------------------------------------------------------------------
# Low-level send / receive
# ---------------------------------------------------------------------------
 
def send_cmd(sock: socket.socket, cmd: bytes, label: str = "") -> bytes:
    """Send *cmd* over *sock* and return the full response frame."""
    sock.sendall(cmd)
    response = sock.recv(256)
    print(f"  [{label}] TX: {binascii.hexlify(cmd).decode()}")
    print(f"  [{label}] RX: {binascii.hexlify(response).decode()}")
    return response
 
 
def parse_fc03_data(response: bytes) -> bytes | None:
    """
    Extract the data bytes from an FC03 Modbus TCP response.
 
    Response layout:
      [TID:2][PID:2][Len:2][UnitID:1][FC:1][ByteCount:1][Data:N]
    Returns the data bytes, or None if the response looks malformed.
    """
    if len(response) < 9:
        return None
    byte_count = response[8]
    data = response[9: 9 + byte_count]
    if len(data) < byte_count:
        return None
    return data
 
 
# ---------------------------------------------------------------------------
# Polling helpers
# ---------------------------------------------------------------------------
 
def wait_for_activation(sock: socket.socket, timeout: float = 10.0) -> bool:
    """Poll until gSTA == 0x03 (activation complete)."""
    print("\n  Waiting for activation to complete...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = send_cmd(sock, CMD_READ_STATUS_1REG, "poll-act")
        data = parse_fc03_data(resp)
        if data and len(data) >= 2:
            gripper_status = data[0]   # GRIPPER STATUS byte
            g_sta = (gripper_status >> 4) & 0x03   # bits 5:4
            print(f"    gSTA = {g_sta}")
            if g_sta == 0x03:
                print("  Activation complete.")
                return True
        time.sleep(0.1)
    print("  WARNING: Activation timed out.")
    return False
 
 
def wait_for_motion(sock: socket.socket, timeout: float = 10.0):
    """Poll until gOBJ != 0x00 (fingers stopped moving)."""
    print("  Waiting for motion to complete...")
    obj_labels = {
        0x01: "object detected while opening",
        0x02: "object detected while closing",
        0x03: "at requested position / no object",
    }
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = send_cmd(sock, CMD_READ_STATUS_3REG, "poll-mot")
        data = parse_fc03_data(resp)
        if data and len(data) >= 6:
            gripper_status = data[0]
            g_obj = (gripper_status >> 6) & 0x03   # bits 7:6
            position  = data[4]                     # gPO  0x00=open, 0xFF=closed
            current   = data[5]                     # gCU  x10 mA
            print(f"    gOBJ={g_obj}  pos={position}/255  current={current*10}mA")
            if g_obj != 0x00:
                print(f"  Motion done – {obj_labels.get(g_obj, '?')}")
                return g_obj
        time.sleep(0.05)
    print("  WARNING: Motion timed out.")
    return None
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    print(f"Connecting to UR10 at {ROBOT_IP}:{MODBUS_PORT} …")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(TIMEOUT)
        sock.connect((ROBOT_IP, MODBUS_PORT))
        print("Connected.\n")
 
        # ------------------------------------------------------------------
        # Step 1: Activation sequence (clear rACT, then set rACT)
        # ------------------------------------------------------------------
        print("--- Step 1: Activate gripper ---")
        send_cmd(sock, CMD_CLEAR_ACTIVATE, "clear rACT")
        time.sleep(0.01)
 
        send_cmd(sock, CMD_ACTIVATE, "set rACT")
        time.sleep(0.01)
 
        wait_for_activation(sock)
 
        # ------------------------------------------------------------------
        # Step 2: Continuous open/close loop
        # ------------------------------------------------------------------
        print("\n--- Starting open/close loop (Ctrl+C to stop) ---\n")
        try:
            while True:
                print(">>> Close gripper")
                send_cmd(sock, CMD_CLOSE, "close")
                wait_for_motion(sock)
                time.sleep(1.0)
 
                print(">>> Open gripper")
                send_cmd(sock, CMD_OPEN, "open")
                wait_for_motion(sock)
                time.sleep(1.0)
 
        except KeyboardInterrupt:
            print("\nStopped by user.")
 
    print("Socket closed.")
 
 
if __name__ == "__main__":
    main()