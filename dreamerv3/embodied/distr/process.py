import multiprocessing as mp
import sys

import cloudpickle

from . import utils


class Process:
    initializers = []
    current_name = None

    def __init__(self, fn, *args, name=None, start=False, pass_running=False):
        name = name or fn.__name__
        fn = cloudpickle.dumps(fn)
        inits = cloudpickle.dumps(self.initializers)
        context = mp.get_context()
        self.errqueue = context.SimpleQueue()
        self.process = context.Process(
            target=self._wrapper,
            name=name,
            args=(fn, name, args, utils.get_print_lock(), self.errqueue, inits),
        )
        self.started = False
        start and self.start()

    @property
    def name(self):
        return self.process.name

    @property
    def pid(self):
        return self.process.pid

    @property
    def alive(self):
        return self.process.is_alive()

    @property
    def exitcode(self):
        return self.process.exitcode

    def start(self):
        assert not self.started
        self.started = True
        self.process.start()

    def check(self):
        assert self.started
        if self.process.exitcode not in (None, 0):
            raise self.errqueue.get()

    def join(self, timeout=None):
        self.check()
        self.process.join(timeout)

    def terminate(self):
        if not self.alive:
            return
        self.process.terminate()

    def __repr__(self):
        attrs = ("name", "pid", "started", "exitcode")
        attrs = [f"{k}={getattr(self, k)}" for k in attrs]
        return f"{type(self).__name__}(" + ", ".join(attrs) + ")"

    @staticmethod
    def _wrapper(fn, name, args, lock, errqueue, inits):
        Process.current_name = name
        try:
            for init in cloudpickle.loads(inits):
                init()
            fn = cloudpickle.loads(fn)
            fn(*args)
            sys.exit(0)
        except Exception as e:
            utils.warn_remote_error(e, name, lock)
            errqueue.put(e)
            sys.exit(1)


class StoppableProcess(Process):
    def __init__(self, fn, *args, name=None, start=False):
        self.runflag = mp.get_context().Event()

        def fn2(runflag, *args):
            assert runflag is not None
            context = utils.Context(runflag.is_set)
            fn(context, *args)

        super().__init__(fn2, self.runflag, *args, name=name, start=start)

    def start(self):
        self.runflag.set()
        super().start()

    def stop(self, wait=10):
        self.check()
        if not self.alive:
            return
        self.runflag.clear()
        if wait is True:
            self.join()
        elif wait:
            self.join(wait)
            if self.alive:
                print(f"Terminating process '{self.name}' that did not want to stop.")
                self.terminate()
