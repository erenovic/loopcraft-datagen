from io import BytesIO

import av
import numpy as np
from PIL import Image


class VideoEncoder:
    fps: int
    bytes_io: BytesIO | None
    container: "av.container.output.OutputContainer | None"
    stream: "av.VideoStream | None"

    def __init__(self, fps: int = 20) -> None:
        self.fps = fps
        self.bytes_io = None
        self.container = None
        self.stream = None

    def add_frame(self, frame: Image.Image) -> None:
        # Ensure that the height and width are divisible by 2.
        w, h = frame.size
        if w % 2 or h % 2:
            frame = frame.crop((0, 0, w - (w % 2), h - (h % 2)))
            w, h = frame.size

        # Lazily initialize the container and stream.
        if self.container is None:
            self.bytes_io = BytesIO()
            self.container = av.open(self.bytes_io, "w", "mp4")
            self.stream = self.container.add_stream("libx264", self.fps)
            self.stream.width = w
            self.stream.height = h
            self.stream.pix_fmt = "yuv420p"

        # Add the frame to the stream.
        array = np.asarray(frame.convert("RGB"))
        video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def result(self) -> bytes:
        # Flush the stream.
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()

        return self.bytes_io.getvalue()
