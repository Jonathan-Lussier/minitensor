from contextlib import contextmanager
import minitensor as mt

@contextmanager
def no_grad():
    """Run a block with graph recording turned off."""
    prev = mt.is_grad_enabled()
    mt.set_grad_enabled(False)
    try:
        yield
    finally:
        mt.set_grad_enabled(prev)