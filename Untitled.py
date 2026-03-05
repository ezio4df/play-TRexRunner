# %%
import gymnasium as gym
from gymnasium import spaces
import cv2
import numpy as np
import time
import threading
import asyncio
import json
import uuid
from collections import deque
from typing import Optional, Tuple, Dict, Any
import sys

# Your existing imports
import gi
from pipewire_capture import PortalCapture, CaptureStream, is_available
import websockets
from gi.repository import Gst, GLib

# Initialize GStreamer (once per process)
Gst.init(None)

# %% [markdown]
# ----
#
# # Screen Capture


# %%
class DinoScreenCapture:
    """Handles PipeWire screen capture with dynamic cropping & preprocessing."""

    def __init__(self, crop_region: Optional[Tuple[int, int, int, int]] = None):
        self.crop_region = crop_region  # (x, y, w, h)
        self.frame = None
        self.latest_frame = None
        self.pipeline = None
        self.sink = None
        self.session = None

    def start_capture(self, fd: int, node_id: int, width: int, height: int):
        """Initialize GStreamer pipeline with PipeWire source."""
        pipeline_str = (
            f"pipewiresrc fd={fd} path={node_id} ! "
            "video/x-raw ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name("sink")
        self.sink.connect("new-sample", self._on_new_sample)
        self.pipeline.set_state(Gst.State.PLAYING)

    def _on_new_sample(self, sink):
        """Callback: convert GStreamer buffer → OpenCV frame."""
        sample = sink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = structure.get_value("width")
            height = structure.get_value("height")

            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                frame = np.ndarray(
                    (height, width, 3), buffer=map_info.data, dtype=np.uint8
                ).copy()
                buffer.unmap(map_info)

                # Apply crop if defined
                if self.crop_region:
                    x, y, w, h = self.crop_region
                    frame = frame[y : y + h, x : x + w]

                self.latest_frame = frame
            return Gst.FlowReturn.OK
        return Gst.FlowReturn.ERROR

    def get_frame(self, grayscale: bool = True) -> Optional[np.ndarray]:
        """Get latest preprocessed frame."""
        if self.latest_frame is None:
            return None
        if grayscale:
            return cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2GRAY)
        return self.latest_frame.copy()

    def set_crop_region(self, region: Tuple[int, int, int, int]):
        """Set crop region for subsequent frames."""
        self.crop_region = region

    def stop(self):
        """Cleanup GStreamer pipeline."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.session:
            self.session.close()


# %%
# # == Testing ==
#
# import time
# crop = None

# print("[info]  Opening system screen picker... Select the Chrome Dino window.")
# portal = PortalCapture()
# session = portal.select_window()

# if not session:
#     print("[info]  No window selected.")
#     sys.exit(1)

# print(f"[info]  Stream acquired: {session.width}x{session.height}")

# capture = DinoScreenCapture(crop_region=crop)
# capture.start_capture(
#     fd=session.fd,
#     node_id=session.node_id,
#     width=session.width,
#     height=session.height
# )

# print("[info]  Capture started. Press 'q' to quit.")

# try:
#     prev_time = time.time()
#     while True:
#         # Get grayscale frame
#         frame = capture.get_frame(grayscale=True)

#         if frame is not None:
#             # Calculate FPS
#             current_time = time.time()
#             delta = current_time - prev_time
#             fps = 1 / delta if delta > 0 else 0
#             prev_time = current_time

#             # Convert back to BGR just for display (OpenCV needs color for imshow usually,
#             # though it handles gray fine. Let's stack to 3 channels for visibility)
#             display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

#             # Add info text
#             h, w = frame.shape
#             cv2.putText(display_frame, f"Shape: {w}x{h} | FPS: {fps:.2f}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             if crop:
#                 cv2.putText(display_frame, "Cropped", (10, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#             cv2.imshow("DinoScreenCapture Test", display_frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break

# except KeyboardInterrupt:
#     print("\n[error]  Interrupted")
# finally:
#     capture.stop()
#     session.close()
#     cv2.destroyAllWindows()
#     print("[info]  Cleanup complete")

# %% [markdown]
# ---
#
# # WS Server

# %% [markdown]
# run it in console
# ```js
# (function () {
#     const WS_SERVER = 'ws://localhost:8766';
#     let ws = null;
#     let reconnectTimeout = null;
#     const PING_INTERVAL = 5_000; // 10 seconds
#     let pingInterval = null;
#
#     function connect() {
#         ws = new WebSocket(WS_SERVER);
#
#         ws.onopen = () => {
#             console.log('[RemoteExec] Connected to WebSocket server');
#             startPing();
#         };
#
#         ws.onmessage = async (event) => {
#             try {
#                 const data = JSON.parse(event.data);
#
#                 if (data.type === 'pong') {
#                     // Keep-alive response, nothing to do
#                     return;
#                 }
#
#                 if (data.type === 'execute') {
#                     const code = data.code;
#
#                     // Wrap user code in an async function
#                     let result;
#                     try {
#                         // Use new Function to avoid eval restrictions and enable async
#                         const fn = new Function('return (async () => {' + code + '})()');
#                         result = fn(); // This returns a Promise
#                     } catch (e) {
#                         // Syntax error or function creation failed
#                         sendResult({
#                             type: 'execution_result',
#                             uid: data.uid,
#                             result: null,
#                             error: '[JS Parse Error] ' + (e.stack || e.toString()),
#                         });
#                         return;
#                     }
#
#                     // Handle Promise resolution
#                     Promise.resolve(result)
#                         .then(value => {
#                             // Serialize result safely
#                             try {
#                                 sendResult({
#                                     type: 'execution_result',
#                                     uid: data.uid,
#                                     result: value,
#                                     error: null,
#                                 });
#                             } catch (e) {
#                                 sendResult({
#                                     type: 'execution_result',
#                                     uid: data.uid,
#                                     result: null,
#                                     error: '[Serialize Error] ' + (e.stack || e.toString()),
#                                 });
#                             }
#                         })
#                         .catch(err => {
#                             sendResult({
#                                 type: 'execution_result',
#                                 uid: data.uid,
#                                 error: String(err),
#                                 result: null,
#                             });
#                         });
#                 }
#             } catch (e) {
#                 console.error('[RemoteExec] Error processing message:', e);
#             }
#         };
#
#         ws.onclose = () => {
#             console.log('[RemoteExec] Disconnected from WebSocket server');
#             stopPing();
#             scheduleReconnect();
#         };
#
#         ws.onerror = (error) => {
#             console.error('[RemoteExec] WebSocket error:', error);
#         };
#     }
#
#     function sendResult(payload) {
#         if (ws && ws.readyState === WebSocket.OPEN) {
#             ws.send(JSON.stringify(payload));
#         }
#     }
#
#     function startPing() {
#         stopPing();
#         pingInterval = setInterval(() => {
#             if (ws && ws.readyState === WebSocket.OPEN) {
#                 ws.send(JSON.stringify({
#                     type: 'ping',
#                     uid: crypto.randomUUID(),
#                     timestamp: Date.now()
#                 }));
#             }
#         }, PING_INTERVAL);
#     }
#
#     function stopPing() {
#         if (pingInterval) {
#             clearInterval(pingInterval);
#             pingInterval = null;
#         }
#     }
#
#     function scheduleReconnect() {
#         if (reconnectTimeout) {
#             clearTimeout(reconnectTimeout);
#         }
#         reconnectTimeout = setTimeout(() => {
#             console.log('[RemoteExec] Reconnecting...');
#             connect();
#         }, 3000);
#     }
#
#     // Start connection
#     connect();
# })();
# ```


# %%
class WebSocketServer:
    def __init__(self, host="localhost", port=8766):
        self.host = host
        self.port = port
        self.clients = {}  # {uid: websocket}
        self.client_uid_counter = 0
        self.client_last_seen = {}  # {uid: timestamp}
        self.TIMEOUT = 45  # seconds
        self.msg_callbacks = {}  # {msg_id: callback}
        self.loop = None

    async def handle_client(self, websocket):
        # Assign unique ID
        self.client_uid_counter += 1
        client_uid = self.client_uid_counter
        self.clients[client_uid] = websocket
        self.client_last_seen[client_uid] = time.time()
        print(f"Client {client_uid} connected", flush=True, file=sys.stdout)

        try:
            async for message in websocket:
                try:
                    print(f"From {client_uid}: {message}", flush=True)
                    data = json.loads(message)

                    # validation
                    """
                    {
                      type: "ping" | "pong" | "execute" | "execution_result"
                      uid: str | int
                      timestamp: number (required only for 'ping'/'pong' messages)
                      result: any (required only for 'execution_result' messages | if no error)
                      error: any (required only for 'execution_result' messages | only if error occurs)
                      code: str (required only for 'execute' messages)
                    }
                    """
                    assert isinstance(data, dict), "Data must be a dictionary"
                    assert "type" in data, "Data must contain 'type' field"
                    assert data["type"] in [
                        "ping",
                        "pong",
                        "execute",
                        "execution_result",
                    ], f"Invalid type: {data['type']}"
                    assert "uid" in data, "Data must contain 'uid' field"
                    assert isinstance(
                        data["uid"], (str, int)
                    ), "UID must be string or int"
                    if data["type"] in ["ping", "pong"]:
                        assert (
                            "timestamp" in data
                        ), f"Timestamp required for {data['type']}"
                        assert isinstance(
                            data["timestamp"], (int, float)
                        ), "Timestamp must be numeric"
                    if data["type"] == "execute":
                        assert "code" in data, "Code required for execute"
                        assert isinstance(data["code"], str), "Code must be string"
                    if data["type"] == "execution_result":
                        assert "result" in data, "Result required for execution_result"
                        assert "error" in data, "Error required for execution_result"

                    await self.handle_message(data, client_uid)
                except json.JSONDecodeError:
                    print(f"Invalid JSON from client {client_uid}")
                except AssertionError as e:
                    print(
                        f"Warning: Invalid message format from client {client_uid}: {e}"
                    )
        except websockets.exceptions.ConnectionClosed:
            print(f"Client {client_uid} disconnected")
        finally:
            if client_uid in self.clients:
                del self.clients[client_uid]
                if client_uid in self.client_last_seen:
                    del self.client_last_seen[client_uid]

    async def handle_message(self, data, client_uid):
        msg_type = data.get("type")
        msg_id = data.get("uid")

        # Update last seen time
        self.client_last_seen[client_uid] = time.time()

        if msg_type == "ping":
            await self.send_message(
                data={"uid": msg_id, "type": "pong", "timestamp": time.time()},
                client_uid=client_uid,
            )

        elif msg_type == "execution_result":
            if msg_id in self.msg_callbacks:
                callback = self.msg_callbacks.pop(msg_id)
                asyncio.create_task(callback(data))

        else:
            print("Warn: msg has no callback!")

    async def send_message(self, data, callback=None, client_uid=None):
        client_uids = [client_uid] if client_uid is not None else self.clients.keys()

        for client_uid in client_uids:
            if client_uid not in self.clients:
                continue

            try:
                # Add message ID if not present
                if "uid" not in data:
                    data["uid"] = str(uuid.uuid4())

                # Register callback if provided
                if callback is not None:
                    self.msg_callbacks[data["uid"]] = callback

                await self.clients[client_uid].send(json.dumps(data))
                print(
                    f"Sent message to client {client_uid}: {data}"
                )  # Log sent message
            except websockets.exceptions.ConnectionClosed:
                # Clean up dead connection
                del self.clients[client_uid]
                if client_uid in self.client_last_seen:
                    del self.client_last_seen[client_uid]

    async def __cleanup_dead_connections(self):
        while True:
            current_time = time.time()
            dead_clients = [
                client_uid
                for client_uid, last_seen in self.client_last_seen.items()
                if current_time - last_seen > self.TIMEOUT
            ]

            for client_uid in dead_clients:
                print(f"Client {client_uid} timed out")
                if client_uid in self.clients:
                    await self.clients[client_uid].close()
                    del self.clients[client_uid]
                if client_uid in self.client_last_seen:
                    del self.client_last_seen[client_uid]

            await asyncio.sleep(5)  # Check every 5 seconds

    def start_server(self):
        async def run_server():
            server = await websockets.serve(self.handle_client, self.host, self.port)
            print(f"WebSocket server started on ws://{self.host}:{self.port}")

            # Start cleanup task
            cleanup_task = asyncio.create_task(self.__cleanup_dead_connections())

            await server.wait_closed()

        # Create a new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(run_server())

    async def execute_js(self, code, timeout=0.01):
        # Generate a unique message ID for this execution
        msg_id = str(uuid.uuid4())

        # Create a future to wait for the response
        future = asyncio.Future()

        # Define the callback that will set the result of the future
        async def callback(data):
            if not future.done():
                future.set_result(data)

        # Send the execute message with the callback
        await self.send_message(
            data={"type": "execute", "code": code, "uid": msg_id}, callback=callback
        )

        # Wait for the callback to be called and return the result
        result = await asyncio.wait_for(future, timeout=timeout)
        return result


# %%
# # == Testing ==
# server = WebSocketServer()
# server_thread = threading.Thread(target=lambda: server.start_server(), daemon=True)
# server_thread.start()
# time.sleep(3)

# await server.execute_js("return 2 + 58")


# %%
class Controller:
    is_ducking = False

    @classmethod
    def __keyev(cls, evType, key):
        assert evType in ["keyup", "keydown"]
        assert key in ["ArrowUp", "ArrowDown"]
        code = {
            "ArrowUp": 38,
            "ArrowDown": 40,
        }[key]

        return f"""
            document.body.dispatchEvent(new KeyboardEvent('{evType}', {{
                key: "{key}",
                code: "{key}",
                bubbles: true,
                cancelable: true,
                view: window,
                which: {code},
                keyCode: {code},
            }}))
        """

    @classmethod
    async def reset(cls):
        cls.is_ducking = False
        return await server.execute_js(
            f"""
            if (runnerInstance.playing) runnerInstance.stop()
            runnerInstance.restart()

            const x = document.querySelector('.runner-canvas').getBoundingClientRect()
            // [x, y, width, height]
            return [abc.x, abc.y + (window.outerHeight - window.innerHeight), abc.width, abc.height]
        """
        )

    @classmethod
    async def jump(cls):
        await server.execute_js(
            f"""
            if ({int(cls.is_ducking)})
                requestAnimationFrame(() => {cls.__keyev('keyup', 'ArrowDown')});
            requestAnimationFrame(() => {cls.__keyev('keydown', 'ArrowUp')});
            requestAnimationFrame(() => {cls.__keyev('keyup', 'ArrowUp')});
            return true
        """
        )
        cls.is_ducking = False

    @classmethod
    async def duck(cls):
        await server.execute_js(
            f"""
            if ({int(not cls.is_ducking)})
                requestAnimationFrame(() => {cls.__keyev('keydown', 'ArrowDown')});
            return true
        """
        )
        cls.is_ducking = True

    @classmethod
    async def noop(cls):
        await server.execute_js(
            f"""
            if ({int(cls.is_ducking)})
                requestAnimationFrame(() => {cls.__keyev('keyup', 'ArrowDown')});
            return true
        """
        )
        cls.is_ducking = False


# %%
# == Testing ==
# await Controller.jump()
# await Controller.duck()
# await Controller.noop()

# %%
# == Testing ==

import time

resp = await Controller.reset()
assert resp["error"] is None
crop = resp["result"]

print("[info]  Opening system screen picker... Select the Chrome Dino window.")
portal = PortalCapture()
session = portal.select_window()

if not session:
    print("[info]  No window selected.")
    sys.exit(1)

print(f"[info]  Stream acquired: {session.width}x{session.height}")

capture = DinoScreenCapture(crop_region=crop)
capture.start_capture(
    fd=session.fd, node_id=session.node_id, width=session.width, height=session.height
)

print("[info]  Capture started. Press 'q' to quit.")

try:
    prev_time = time.time()
    while True:
        # Get grayscale frame
        frame = capture.get_frame(grayscale=True)

        if frame is not None:
            # Calculate FPS
            current_time = time.time()
            delta = current_time - prev_time
            fps = 1 / delta if delta > 0 else 0
            prev_time = current_time

            # Convert back to BGR just for display (OpenCV needs color for imshow usually,
            # though it handles gray fine. Let's stack to 3 channels for visibility)
            display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Add info text
            h, w = frame.shape
            cv2.putText(
                display_frame,
                f"Shape: {w}x{h} | FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            if crop:
                cv2.putText(
                    display_frame,
                    "Cropped",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("DinoScreenCapture Test", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("r"):
            await Controller.reset()

        if key == 82:  # ArrowUp
            await Controller.jump()

        if key == 84:  #  ArrowDown
            await Controller.duck()

        if key == 83:  #  ArrowRight
            await Controller.noop()

except KeyboardInterrupt:
    print("\n[error]  Interrupted")
finally:
    capture.stop()
    session.close()
    cv2.destroyAllWindows()
    print("[info]  Cleanup complete")

# %%
