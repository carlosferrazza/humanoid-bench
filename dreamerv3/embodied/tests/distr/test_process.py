import multiprocessing as mp
import pathlib
import sys
import time
import traceback

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))

import embodied
import pytest


class TestProcess:
    def test_terminate(self):
        def fn():
            while True:
                time.sleep(0.01)

        worker = embodied.distr.Process(fn, start=True)
        worker.terminate()
        worker.join()

    def test_stop(self):
        def fn(context, q):
            q.put("start")
            while context.running:
                time.sleep(0.01)
            q.put("stop")

        q = mp.get_context().SimpleQueue()
        worker = embodied.distr.StoppableProcess(fn, q)
        worker.start()
        worker.stop()
        assert q.get() == "start"
        assert q.get() == "stop"

    def test_exitcode(self):
        worker = embodied.distr.Process(lambda: None)
        assert worker.exitcode is None
        worker.start()
        worker.join()
        assert worker.exitcode == 0

    def test_exception(self):
        def fn1234(q):
            q.put(42)
            raise KeyError("foo")

        q = mp.get_context().SimpleQueue()
        worker = embodied.distr.Process(fn1234, q, start=True)
        q.get()
        time.sleep(0.2)
        assert not worker.alive
        assert worker.exitcode == 1
        with pytest.raises(KeyError) as info:
            worker.check()
        assert repr(info.value) == "KeyError('foo')"
        tb = "".join(traceback.format_exception(info.value))
        assert "KeyError: 'foo'" in tb
        if sys.version_info.minor >= 11:
            assert "Traceback" in tb
            assert " File " in tb
            assert "fn1234" in tb
