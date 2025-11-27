import os
import queue  # Import the queue module for exception type hint
import signal
import multiprocessing
from functools import wraps
from typing import Any, Callable, Iterator, Optional

# --- Top-level helper for multiprocessing timeout ---
# This function MUST be defined at the top level to be pickleable
def _mp_target_wrapper(target_func: Callable, mp_queue: multiprocessing.Queue, args: tuple, kwargs: dict[str, Any]):
    """
    Internal wrapper function executed in the child process.
    Calls the original target function and puts the result or exception into the queue.
    """
    try:
        result = target_func(*args, **kwargs)
        mp_queue.put((True, result))  # Indicate success and put result
    except Exception as e:
        # Ensure the exception is pickleable for the queue
        try:
            import pickle

            pickle.dumps(e)  # Test if the exception is pickleable
            mp_queue.put((False, e))  # Indicate failure and put exception
        except (pickle.PicklingError, TypeError):
            # Fallback if the original exception cannot be pickled
            mp_queue.put((False, RuntimeError(f"Original exception type {type(e).__name__} not pickleable: {e}")))

# Renamed the function from timeout to timeout_limit
def timeout_limit(seconds: float, use_signals: bool = False):
    """
    Decorator to add a timeout to a function.

    Args:
        seconds: The timeout duration in seconds.
        use_signals: (Deprecated)  This is deprecated because signals only work reliably in the main thread
                     and can cause issues in multiprocessing or multithreading contexts.
                     Defaults to False, which uses the more robust multiprocessing approach.

    Returns:
        A decorated function with timeout.

    Raises:
        TimeoutError: If the function execution exceeds the specified time.
        RuntimeError: If the child process exits with an error (multiprocessing mode).
        NotImplementedError: If the OS is not POSIX (signals are only supported on POSIX).
    """

    def decorator(func):
        if use_signals:
            if os.name != "posix":
                raise NotImplementedError(f"Unsupported OS: {os.name}")
            # Issue deprecation warning if use_signals is explicitly True
            print(
                "WARN: The 'use_signals=True' option in the timeout decorator is deprecated. \
                Signals are unreliable outside the main thread. \
                Please use the default multiprocessing-based timeout (use_signals=False)."
            )

            @wraps(func)
            def wrapper_signal(*args, **kwargs):
                def handler(signum, frame):
                    # Update function name in error message if needed (optional but good practice)
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds (signal)!")

                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                # Use setitimer for float seconds support, alarm only supports integers
                signal.setitimer(signal.ITIMER_REAL, seconds)

                try:
                    result = func(*args, **kwargs)
                finally:
                    # Reset timer and handler
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
                return result

            return wrapper_signal
        else:
            # --- Multiprocessing based timeout (existing logic) ---
            @wraps(func)
            def wrapper_mp(*args, **kwargs):
                q = multiprocessing.Queue(maxsize=1)
                process = multiprocessing.Process(target=_mp_target_wrapper, args=(func, q, args, kwargs))
                process.start()
                process.join(timeout=seconds)

                if process.is_alive():
                    process.terminate()
                    process.join(timeout=0.5)  # Give it a moment to terminate
                    if process.is_alive():
                        print(f"Warning: Process {process.pid} did not terminate gracefully after timeout.")
                    # Update function name in error message if needed (optional but good practice)
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds (multiprocessing)!")

                try:
                    success, result_or_exc = q.get(timeout=0.1)  # Small timeout for queue read
                    if success:
                        return result_or_exc
                    else:
                        raise result_or_exc  # Reraise exception from child
                except queue.Empty as err:
                    exitcode = process.exitcode
                    if exitcode is not None and exitcode != 0:
                        raise RuntimeError(
                            f"Child process exited with error (exitcode: {exitcode}) before returning result."
                        ) from err
                    else:
                        # Should have timed out if queue is empty after join unless process died unexpectedly
                        # Update function name in error message if needed (optional but good practice)
                        raise TimeoutError(
                            f"Operation timed out or process finished unexpectedly without result "
                            f"(exitcode: {exitcode})."
                        ) from err
                finally:
                    q.close()
                    q.join_thread()

            return wrapper_mp

    return decorator