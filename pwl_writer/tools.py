"""This module defines a PWL object menat to create pwl files in terms of pulses and transitions, considering time durations instead of time instants.
"""

from numbers import Real
import numpy
from typing import Optional, List,  Dict, Callable, TypeVar


class PrecisionError(Exception):
    """Exception class for precision related errors."""


def _exp_edge_func(tau: float, t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:
    """Private function that generates an exponential function passing trough 2 fixed points."""

    A = (f1*numpy.exp(t1/tau) - f2*numpy.exp(t2/tau)) / \
        (numpy.exp(t1/tau) - numpy.exp(t2/tau))
    B = (f1 - f2)/(numpy.exp(-t1/tau) - numpy.exp(-t2/tau))

    def f(t: float) -> float:
        return A + B*numpy.exp(-t/tau)

    return f


def _sin_edge_func(t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:
    """Private function that generates a half sine function passing trough 2 fixed points."""

    fm = (f1+f2)/2
    tm = (t1+t2)/2
    T = 2*(t2-t1)
    w = 2*numpy.pi/T
    phi = w*tm
    A = f2-fm

    def f(t: float) -> float:
        return fm + A*numpy.sin(w*t - phi)

    return f


def _smoothstep_edge_func(t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:
    """Private function that generates a smoothstep polynomial passing trough 2 fixed points."""

    Am = numpy.array([[1, t1, t1**2, t1**3],
                      [1, t2, t2**2, t2**3],
                      [0, 1, 2*t1, 3*t1**2],
                      [0, 1, 2*t2, 3*t2**2]])
    Bm = numpy.array([f1, f2, 0, 0])
    A, B, C, D = numpy.linalg.solve(Am, Bm)

    def f(t):
        return A + B*t + C*t**2 + D*t**3

    return f


class PWL():
    """Class defining a PWL object used to create a pwl file."""

    _dict_of_objects: Dict = {}

    def __init__(self, t_step: float = 1e-9, name: Optional[str] = None, verbose: bool = False) -> None:
        """Initializer for the PWL class.

        PWL objects are a fancy way of storing a list of time coordinates and another of dependent coordinates. It also stores some additional metadata to provide syntactic sugar when adding pulses and transitions to the PWL object.

        Parameters
        ----------
        `t_step` : float
            Default timestep for the different types of pulses and transitions. Should be strictly positive. Defaults to `1e-9`.
        `name` : str, optional
            Name of the PWL object used for verbose output. If set to `None`, auto generates based on the names of already created objects. Defaults to `None`.
        `verbose` : bool, optional
            Flag to control if verbose output should be printed or not. Defaults to `False`.

        Raises
        ------
        `TypeError`
            Raised if any of the arguments has an invalid type.
        `ValueError`
            Raised if `t_step` is non positive or if `name` is an empty string.
        """

        # Check for nullable arguments
        if name is None:
            i: int = 0
            while f"pwl_{i}" in PWL._dict_of_objects:
                i += 1
            name = f"pwl_{i}"

        # Check type of arguments
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should be a real number but has type '{type(t_step).__name__}'.")
        if not isinstance(name, str):
            raise TypeError(
                f"Argument 'name' should either be a string or be None but has type '{type(name).__name__}'.")
        if not isinstance(verbose, bool):
            raise TypeError(
                f"Argument 'verbose' should be a boolean but has type '{type(verbose).__name__}'.")

        # Check value of arguments
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}.")
        if not name:
            raise ValueError("Argument 'name' should not be empty.")

        # Actual function
        self._t_list: list[float] = []
        self._x_list: list[float] = []
        self._t_step: float = t_step
        self._name: str = name
        self._verbose: bool = verbose

        if name in PWL._dict_of_objects:
            raise ValueError(f"Name '{name}' already in use.")

        PWL._dict_of_objects[name] = self

    def __str__(self) -> str:
        """String representation of PWL objects."""
        return f"{self.name}: PWL object with {len(self.t_list)} points and duration of {self.t_list[-1]} seconds"

    @property
    def t_list(self) -> List[float]:
        """This read-only property is the list of all time coordinates."""

        return self._t_list

    @property
    def x_list(self) -> List[float]:
        """This read-only property is the list of all dependent coordinates."""

        return self._x_list

    @property
    def t_step(self) -> float:
        """This property is the default timestep for the different types of pulses and transitions.

        Raises
        ------
        `TypeError`
            Raised if the passed value is not a real number.
        `ValueError`
            Raised if the passed value is not strictly positive.
        """

        return self._t_step

    @t_step.setter
    def t_step(self, new_t_step: float) -> None:
        """Setter for the `t_step` property."""

        if not isinstance(new_t_step, Real):
            raise TypeError(
                f"Attribute 't_step' should be a real number but has type '{type(new_t_step).__name__}'.")
        if new_t_step <= 0:
            raise ValueError(
                f"Attribute 't_step' should be strictly positive but has value of {new_t_step}.")

        self._t_step = new_t_step

    @property
    def name(self) -> str:
        """This property is the name of the PWL object used for verbose output.

        Raises
        ------
        `TypeError`
            Raised if the passed name is not a string.
        `ValueError`
            Raised if the passed name is either empty or already in use.
        """

        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """Setter for the `name` property."""

        if not isinstance(new_name, str):
            raise TypeError(
                f"Attribute 'name' should be a string but has type '{type(new_name).__name__}'.")
        if not new_name:
            raise ValueError("Attribute 'name' should not be an empty string.")

        if new_name in PWL._dict_of_objects:
            raise ValueError(f"Name '{new_name}' already in use.")

        PWL._dict_of_objects.pop(self._name)
        PWL._dict_of_objects[new_name] = self
        self._name = new_name

    @property
    def verbose(self) -> bool:
        """This property is a flag indicating if verbose output should be printed or not.

        Raises
        ------
        `TypeError`
            Raised if the passed flag is not a boolean.
        `ValueError`
        """

        return self._verbose

    @verbose.setter
    def verbose(self, new_verbose: bool) -> None:
        """Setter for the `verbose` property."""

        if not isinstance(new_verbose, bool):
            raise TypeError(
                f"Attribute 'verbose' should be a boolean but has type '{type(new_verbose).__name__}'.")
        self._verbose = new_verbose

    def _add(self, t: float, x: float) -> None:
        """Private method to add (t, x) points to the PWL object."""

        if len(self._t_list) >= 1 and t <= self._t_list[-1]:
            raise PrecisionError(
                f"Internal Python rounding caused the time coordinates to not be strictly increasing when adding points to {self._name}.")

        if len(self._t_list) == len(self._x_list) < 2:
            self._t_list.append(t)
            self._x_list.append(x)
        else:
            self._add_with_redundancy_check(x, t)

    def _add_with_redundancy_check(self, x: float, t: float):
        """Private method to add (t, x) points to the PWL object. Makes sure no 3 consecutive points are colinear."""

        t_n_1 = self._t_list[-1]
        t_n_2 = self._t_list[-2]

        x_n_1 = self._x_list[-1]
        x_n_2 = self._x_list[-2]

        last_m = (x_n_1 - x_n_2)/(t_n_1 - t_n_2)
        new_m = (x - x_n_1)/(t - t_n_1)

        if last_m == new_m:
            self._t_list[-1] = t
            self._x_list[-1] = x
        else:
            self._t_list.append(t)
            self._x_list.append(x)

    def hold(self, duration: float) -> None:
        """Method that takes the last dependent value and holds it for a given duration.

        If the PWL object is empty, adds the (0, 0) point.

        Parameters
        ----------
        `duration` : float
            Duration to hold previous value for. Should be strictly positive.

        Raises
        ------
        `TypeError`
            Raised if `duration` is not a real number.
        `ValueError`
            Raised if `duration` is not strictly positive.
        `PrecisionError`
            Raised if numerical noise causes the time coordinates to not be strictly increasing.
        """

        # Check type of arguments
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")

        # Check value of arguments
        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")

        # Actual function
        if self._verbose:
            print(f"{self._name}: Adding hold with duration of {duration}.")

        if len(self._t_list) == len(self._x_list) == 0:
            if self._verbose:
                print("    Empty PWL object. Adding initial (0, 0) point.")
            self._add(0, 0)

        last_t = self._t_list[-1]
        last_x = self._x_list[-1]

        self._add(last_t+duration, last_x)

    def square_pulse(self, value: float, duration: float, t_step: Optional[float] = None) -> None:
        """Method that generates a square pulse with given amplitude and duration.

        If `duration` is less than or equal to `t_step`, replace the square pulse by a linear edge with `lin_edge` going from the previous value to the desired amplitude with a duration of `t_step`.

        Parameters
        ----------
        `value` : float
            Amplitude of the square pulse.
        `duration` : float
            Duration of the pulse. Should be strictly positive.
        `t_step` : float, optional
            Rising or falling time at start of pulse. Should be strictly positive. If set to `None`, uses the `t_step` property. Defaults to 'None'.

        Raises
        ------
        `TypeError`
            Raised if any argument has an invalid type.
        `ValueError`
            Raised if either `duration` or `t_step` is not strictly positive.
        `PrecisionError`
            Raised if numerical noise causes the time coordinates to not be strictly increasing.

        See Also
        --------
        `lin_edge` : Method that generates a linear transition from the previous dependent value to a given target with given duration.
        """

        # Check for nullable arguments
        if t_step is None:
            t_step = self._t_step

        # Check type of arguments
        if not isinstance(value, Real):
            raise TypeError(
                f"Argument 'value' should be a real number but has type '{type(value).__name__}'.")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should either be a real number or be None but has type '{type(t_step).__name__}'.")

        # Check value of arguments
        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}.")

        # Actual function
        if self._verbose:
            print(f"{self._name}: Adding square pulse with value of {value}, duration of {duration} and time step of {t_step}.")

        if duration <= t_step:
            if self._verbose:
                print(
                    f"{self._name}: Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear edge.")
            self._lin_edge(value, t_step, 1)
            return

        if len(self._t_list) == len(self._x_list) == 0:
            self._add(0, value)
            last_t = 0
        else:
            last_t = self._t_list[-1]
            self._add(last_t+t_step, value)

        self._add(last_t+duration, value)

    def sawtooth_pulse(self, start: float, end: float, duration: float, t_step: Optional[float] = None) -> None:
        """Method that generates a sawtooth pulse with given starting and ending amplitudes and duration.

        If we call `t0` the instant when the pulse begins, the pulse follows the equation `f(t) = A + B*t` with `A` and `B` chosen such that `f(t0 + t_step) = start` and `f(t_0 + duration) = end`.

        If `duration` is less than or equal to `t_step`, replace the sawtooth pulse by a linear edge with `lin_edge` going from the previous value to the desired ending amplitude with a duration of `t_step`.

        Parameters
        ----------
        `start` : float
            Amplitude at the start of the pulse.
        `end` : float
            Amplitude at the end of the pulse.
        `duration` : float
            Duration of the pulse. Should be strictly positive.
        `t_step` : float, optional
            Rising or falling time at start of pulse. Should be strictly positive. If set to `None`, uses the `t_step` property. Defaults to `None`.

        Raises
        ------
        `TypeError`
            Raised if any argument has an invalid type.
        `ValueError`
            Raised if either `duration` or `t_step` is not strictly positive.
        `PrecisionError`
            Raised if numerical noise causes the time coordinates to not be strictly increasing.

        See Also
        --------
        `lin_edge` : Method that generates a linear transition from the previous dependent value to a given target with given duration.
        """

        # Check for nullable arguments
        if t_step is None:
            t_step = self._t_step

        # Check type of arguments
        if not isinstance(start, Real):
            raise TypeError(
                f"Argument 'start' should be a real number but has type '{type(start).__name__}'.")
        if not isinstance(end, Real):
            raise TypeError(
                f"Argument 'end' should be a real number but has type '{type(end).__name__}'.")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should either be a real number or be None but has type '{type(t_step).__name__}'.")

        # Check value of arguments
        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}.")

        # Actual function
        if self._verbose:
            print(f"{self._name}: Adding sawtoth pulse from {start} to {end} with duration of {duration} and time step of {t_step}.")

        if duration <= t_step:
            if self._verbose:
                print(
                    f"{self._name}: Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear edge.")
            self._lin_edge(end, t_step, 1)
            return

        if len(self._t_list) == len(self._x_list) == 0:
            self._add(0, start)
            last_t = 0
        else:
            last_t = self._t_list[-1]
            self._add(last_t+t_step, start)

        self._add(last_t+duration, end)

    def lin_edge(self, target: float, duration: float) -> None:
        """Generates a linear transition from the previous dependent value to a given targer with a given duration.
        
        If we call `t0` the instant when the pulse begins and `x0` the value the signal assumed just before that, the pulse follows the equation `f(t) = A + B*t` with `A` and `B` chosen such that `f(t0) = x0` and `f(t_0 + duration) = target`.

        Parameters
        ----------
        `target` : float
            Value to transition towards.
        `duration` : float
            Duration of the transition. Should be strictly positive.

        Raises
        ------
        `TypeError`
            Raised if any argument has an invalid type.
        `ValueError`
            Raised if either `duration` or `t_step` is not strictly positive.
        `PrecisionError`
            Raised if numerical noise causes the time coordinates to not be strictly increasing.
        """

        # Check type of arguments
        if not isinstance(target, Real):
            raise TypeError(
                f"Argument 'target' should be a real number but has type '{type(target).__name__}'.")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")

        # Check value of arguments
        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")

        # Actual function
        self._lin_edge(target, duration, 0)

    def _lin_edge(self, target: float, duration: float, n: int) -> None:
        """Private method to genarate linear transitions. Prints verbose output with added indendation if called inside other methods."""

        if self._verbose:
            if n == 0:
                print(
                    f"{self._name}: Adding linear edge with target of {target} and duration of {duration}.")
            else:
                print(
                    n*"    "+f"Adding linear edge with target of {target} and duration of {duration}.")

        if len(self._t_list) == len(self._x_list) == 0:
            if self._verbose:
                print((n+1)*"    "+"Empty PWL object. Adding initial (0, 0) point.")
            self._add(0, 0)

        last_t = self._t_list[-1]
        self._add(last_t+duration, target)

    def exp_edge(self, target: float, duration: float, tau: float, t_step: Optional[float] = None) -> None:

        # Check for nullable arguments
        if t_step is None:
            t_step = self._t_step

        # Check type of arguments
        if not isinstance(target, Real):
            raise TypeError(
                f"Argument 'target' should be a real number but has type '{type(target).__name__}'.")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")
        if not isinstance(tau, Real):
            raise TypeError(
                f"Argument 'tau' should be a real number but has type '{type(tau).__name__}'.")
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should either be a real number or be None but has type '{type(t_step).__name__}'.")

        # Check value of arguments
        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if tau == 0:
            raise ValueError("Argument 'tau' should be non zero.")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}.")

        # Actual function
        if self._verbose:
            print(f"{self._name}: Adding exponential edge with target of {target}, time constant of {tau}, duration of {duration} and time step of {t_step}.")

        if duration <= t_step:
            if self._verbose:
                print(
                    f"    Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear edge.")
            self._lin_edge(target, t_step, 2)
            return

        if len(self._t_list) == len(self._x_list) == 0:
            if self._verbose:
                print("    Empty PWL object. Adding initial (0, 0) point.")
            self._add(0, 0)

        last_t = self._t_list[-1]
        last_x = self._x_list[-1]

        f = _exp_edge_func(tau=tau, t1=last_t, t2=last_t +
                           duration, f1=last_x, f2=target)

        for t in numpy.arange(last_t+t_step, last_t+duration, t_step):
            self._add(t, f(t))

        self._add(last_t+duration, target)

    def sin_edge(self, target: float, duration: float, t_step: Optional[float] = None) -> None:

        # Check for nullable arguments
        if t_step is None:
            t_step = self._t_step

        # Check type of arguments
        if not isinstance(target, Real):
            raise TypeError(
                f"Argument 'target' should be a real number but has type '{type(target).__name__}'.")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should either be a real number or be None but has type '{type(t_step).__name__}'.")

        # Check value of arguments
        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}.")

        # Actual function
        if self._verbose:
            print(f"{self._name}: Adding sinusoidal edge with target of {target}, duration of {duration} and time step of {t_step}.")

        if duration <= t_step:
            if self._verbose:
                print(
                    f"    Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear edge.")
            self._lin_edge(target, t_step, n=2)
            return

        if len(self._t_list) == len(self._x_list) == 0:
            if self._verbose:
                print("    Empty PWL object. Adding initial (0, 0) point.")
            self._add(0, 0)

        last_t = self._t_list[-1]
        last_x = self._x_list[-1]

        f = _sin_edge_func(t1=last_t, t2=last_t+duration, f1=last_x, f2=target)

        for t in numpy.arange(last_t+t_step, last_t+duration, t_step):
            self._add(t, f(t))

        self._add(last_t+duration, target)

    def smoothstep_edge(self, target: float, duration: float, t_step: Optional[float] = None) -> None:

        # Check for nullable arguments
        if t_step is None:
            t_step = self._t_step

        # Check type of arguments
        if not isinstance(target, Real):
            raise TypeError(
                f"Argument 'target' should be a real number but has type '{type(target).__name__}'.")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should either be a real number or be None but has type '{type(t_step).__name__}'.")

        # Check value of arguments
        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}.")

        # Actual function
        if self._verbose:
            print(f"{self._name}: Adding smoothstep edge with target of {target}, duration of {duration} and time step of {t_step}.")

        if duration <= t_step:
            if self._verbose:
                print(
                    f"    Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear edge.")
            self._lin_edge(target, t_step, n=2)
            return

        if len(self._t_list) == len(self._x_list) == 0:
            if self._verbose:
                print("    Empty PWL object. Adding initial (0, 0) point.")
            self._add(0, 0)

        last_t = self._t_list[-1]
        last_x = self._x_list[-1]

        f = _smoothstep_edge_func(
            t1=last_t, t2=last_t+duration, f1=last_x, f2=target)

        for t in numpy.arange(last_t+t_step, last_t+duration, t_step):
            self._add(t, f(t))

        self._add(last_t+duration, target)

    def write(self, filename: str, precision: int = 10) -> None:

        # Check type of arguments
        if not isinstance(filename, str):
            raise TypeError(
                f"Argument 'filename' should be a string but has type '{type(filename).__name__}'.")
        if not isinstance(precision, int):
            raise TypeError(
                f"Argument 'precision' should be an integer but has type '{type(precision).__name__}'.")

        # Check value of arguments
        if precision <= 0:
            raise ValueError(
                f"Argument 'precision' should be strictly positive but has value of {precision}.")

        # Actual function
        if self._verbose:
            print(f"{self._name}: Writing PWL file to {filename}.")

        t_list = self._t_list
        x_list = self._x_list

        with open(filename, "w") as file:
            ti_str = numpy.format_float_scientific(
                t_list[0], precision-1, unique=False, sign=False)
            xi_str = numpy.format_float_scientific(
                x_list[0], precision-1, unique=False, sign=True)
            file.write(f"{ti_str}    {xi_str}\n")
            last_t = ti_str
            for ti, xi in zip(t_list[1:], x_list[1:]):
                ti_str = numpy.format_float_scientific(
                    ti, precision-1, unique=False, sign=False)
                xi_str = numpy.format_float_scientific(
                    xi, precision-1, unique=False, sign=True)
                if ti_str == last_t:
                    raise PrecisionError(
                        "The chosen precision level caused the written time coordinates to not be strictly increasing.")
                file.write(f"{ti_str}    {xi_str}\n")
                last_t = ti_str


if __name__ == "__main__":
    pwl = PWL(verbose=True, t_step=0.1)
    pwl.hold(1)
    pwl.sin_edge(1, 1)
    pwl.hold(1)
    pwl.smoothstep_edge(0, 1)
    pwl.square_pulse(1, 1)
    pwl.lin_edge(0, 1)
    pwl.sin_edge(1, 0.01)
    pwl.hold(1)
    print(pwl)
