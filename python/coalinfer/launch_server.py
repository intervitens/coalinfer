"""Launch the inference server."""

import os
import sys

from coalinfer.srt.server import launch_server
from coalinfer.srt.server_args import prepare_server_args
from coalinfer.srt.utils import kill_child_process

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_server(server_args)
    except Exception as e:
        raise e
    finally:
        kill_child_process(os.getpid(), including_parent=False)
