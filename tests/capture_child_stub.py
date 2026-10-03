"""Deliberately unhealthy protocol peer for the capture supervisor's process tests."""
import base64
import json
import sys
import time

mode = sys.argv[1]
if mode == "worker_hang":
    from server import capture_worker, coreaudio
    output = sys.stdout
    selection = {"kind": "microphone", "endpoint_id": "test-endpoint", "name": "Test",
                 "index": 0, "follow_default": False}
    coreaudio.list_endpoints = lambda _: [dict(selection, loopback=False)]
    class HungEndpoint:
        def __init__(self, *args): pass
        def __enter__(self):
            output.write(json.dumps({"type": "ready", "target": selection}) + "\n")
            output.flush()
            while True:
                time.sleep(.1)
        def __exit__(self, *args): pass
    coreaudio.EndpointStream = HungEndpoint
    capture_worker.sys.platform = "win32"  # The endpoint itself is a double, on every host.
    sys.argv = [sys.argv[0], json.dumps(selection)]
    capture_worker.main()
    raise SystemExit(0)
if mode == "crash":
    raise SystemExit(3)
if mode == "malformed":
    print("not JSON", flush=True)
elif mode == "oversized":
    print("x" * 18000, flush=True)
else:
    print(json.dumps({"type": "ready", "target": {
        "kind": "microphone", "endpoint_id": "test-endpoint", "name": "Test input",
        "index": 0, "follow_default": False}}), flush=True)
    if mode == "bad_audio":
        print(json.dumps({"type": "audio", "data": "AA=="}), flush=True)
    elif mode == "overflow":
        for _ in range(80):
            print(json.dumps({"type": "audio",
                             "data": base64.b64encode(bytes(2048)).decode()}), flush=True)
    elif mode == "nan_audio":
        import struct
        print(json.dumps({"type": "audio",
                         "data": base64.b64encode(struct.pack("<512f", *([float("nan")] * 512))).decode()}),
              flush=True)

if mode == "hang":
    while True:
        time.sleep(.1)  # Ignore stop, just as a blocked native driver would.
else:
    # EOF models the parent's death; a normal stop releases an otherwise idle child.
    sys.stdin.readline()
