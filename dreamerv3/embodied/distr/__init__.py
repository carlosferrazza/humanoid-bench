import multiprocessing as mp

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

from .client import Client
from .thread import Thread, StoppableThread
from .process import Process, StoppableProcess
from .utils import run, warn_remote_error, terminate_subprocesses
from .server import Server
from .proc_server import ProcServer
from .sockets import NotAliveError, RemoteError, ProtocolError
