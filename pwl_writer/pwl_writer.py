"""
*by Victor Sabiá P. Carpes*

Tested on python version `3.6.6` with numpy version `1.19.5`. Type stubs for this numpy version for mypy checking can be found [here](https://github.com/numpy/numpy-stubs).

----

This package defines a class `PWL` to generate objects that represent time dependent signals `x(t)` that need to be coded in a PWL file. Those objects are built using little components such as square pulses and sawtooth pulses that can be chained together.

The motivations for this package are the nuisances of writing PWL files by hand. To properly explain this, let's discuss how PWL files work.

PWL stands for piecewise linear. A PWL file is a way to represent a time dependent signal (referenced by `x(t)` from now on) for simulation softwares such as LTspice and Cadence Virtuoso. In it, points of the form `(t, x)` are stored. During simulation, those points are interpolated with first degree polynomials. This poses 2 problems:

1. Due to the linear interpolation, the resulting signal is continuous. This tends to be desirable, but if the intention is moddeling, for example, square pulses, each transition will need 2 points with very close time coordinates to approximate a discontinuous transition. This can get extremely tedious to code out by hand.

2. Each point has an absolute time coordinate with respect to the origin. If the desired signal is for example a series of square pulses with certain durations and for some reason the duration of the first pulse is changed, all the following points will need to have their time coordinates changed as well.

This package solves both problems by providing an abstraction layer. They are solved by the 2 following features:

1. A minimal timestep is defined at the creation of the PWL object that is used to automatically generate all the needed transitions for any discontinous transition.

2. The signal is built using small building blocks (such as square pulse and exponential transition) called events that are defined in terms of durations. That is to say, time is treated in a differential fashion. The time coordinates from a given event are all with respect to the final instant of the previous event. For example, let's assume we want to model a square pulse with amplitude 1 and duration 1 second followed by a downtime at zero for 10 seconds and then another square pulse with the same duration and amplitude. If we change the duration of the first pulse to 2 seconds, the downtime and second pulse will be both  delayed by the 1 second but retain their durations.

Another advantage of using this package is not a feature per se but more a consequence of using a programing language. That advantage is simply that all those events can be added inside for loops, if clauses and functions, allowing for greater flexibility. For example, let's assume we want to control a system that can be in the following states:

* Idle
* Mode 1
* Mode 2

For each state, various control signals need to be at specific values. We could create one `PWL` object for each control signal and define 3 functions that apply all the needed values for the control signals for each state. If we nedded the system to be at mode 1 for 3 seconds, idle for 1 second and at mode 2 for 5 seconds, we could write something like the following:

        mode1_state(3)
        idle_state(1)
        mode2_state(5)

----
"""


from numbers import Real
from typing import Callable, Dict, List, Optional
import numpy

# ----

# = `PrecisionError` =


class PrecisionError(Exception):
    """**Exception class**

    This class defines an exception meant to be raised when any type of rounding or loss of precision that causes the time coordinates of a PWL object to not be strictly increasing.
    """

# ----

# = `PWL` =


class PWL():
    """**Class**

    This class defines an object that represnts a time dependent signal `x(t)`. Those objects can operated on by methods to build, event by event, the desired signal as described on the package introduction.
    """

    __dict_of_objects: Dict[str, 'PWL'] = {}

    # ----

    # = `__init__` =

    def __init__(self, t_step: float, name: Optional[str] = None, verbose: bool = False) -> None:
        """**Method of `PWL` class**

        Summary
        -------
        Initializer for the `PWL` class.

        Arguments
        ---------
        * `t_step` (`float`) : Default timestep for all operations. Should be strictly positive.
        * `name` (`str`, optional) : Name of the `PWL` object used for verbose output printing. Should not be empty. If not set, automatically generates a name based on already taken names.
        * `verbose` (`bool`, optional) : Flag indicating if verbose output should be printed. If not set, defaults to `False`.

        Raises
        ------
        * `TypeError` : Raised if any of the arguments has an invalid type.
        * `ValueError` : Raised if `t_step` is not strictly positive or `name` is empty.
        """

        if name is None:
            i: int = 0
            while f"pwl_{i}" in PWL.__dict_of_objects:
                i += 1
            name = f"pwl_{i}"

        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should be a real number but has type '{type(t_step).__name__}'.")
        if not isinstance(name, str):
            raise TypeError(
                f"Argument 'name' should either be a string but has type '{type(name).__name__}'.")
        if not isinstance(verbose, bool):
            raise TypeError(
                f"Argument 'verbose' should be a boolean but has type '{type(verbose).__name__}'.")

        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}.")
        if not name:
            raise ValueError("Argument 'name' should not be empty.")

        self._t_list: List[float] = []
        self._x_list: List[float] = []
        self._t_step: float = t_step
        self._name: str = name
        self._verbose: bool = verbose

        if name in PWL. __dict_of_objects:
            raise ValueError(f"Name '{name}' already in use.")

        PWL. __dict_of_objects[name] = self

    # ----

    # = `__str__` =

    def __str__(self) -> str:
        """**Method of `PWL` class**

        Summary
        -------
        String representation of `PWL` instances in the form `[name]: PWL object with [# of points] and duration of [total time duration] seconds`.

        Returns
        -------
        * `str`
        """

        return f"{self.name}: PWL object with {len(self.t_list)} points and duration of {self.t_list[-1]} seconds"

    # ----

    # = `t_list` : `list[float]` =

    @property
    def t_list(self) -> List[float]:
        """**Property of `PWL` class**

        Summary
        -------
        Read only property containing all the time coordinates of a `PWL` object.

        Raises
        ------
        * `AttributeError` : Raised if assignment was attempetd.
        """

        return self._t_list

    # ----

    # = `x_list` : `list[float]` =

    @property
    def x_list(self) -> List[float]:
        """**Property of `PWL` class**

        Summary
        -------
        Read only property containing all the dependent coordinates of a `PWL` object.

        Raises
        ------
        * `AttributeError` : Raised if assignment was attempetd.
        """

        return self._x_list

    # ----

    # = `t_step` : `float` =

    @property
    def t_step(self) -> float:
        """**Property of `PWL` class**

        Summary
        -------
        Property defining the default timestep of a `PWL` object.

        Raises
        ------
        * `TypeError` : Raised if the assigned value is not a real number.
        * `ValueError` : Raised if the assigned value is not strictly positive.
        """

        return self._t_step

    @t_step.setter
    def t_step(self, new_t_step: float) -> None:
        if not isinstance(new_t_step, float):
            raise TypeError(
                f"Property 't_step' should be a real number but an object of type '{type(new_t_step).__name__}' was assigned to it.")
        if new_t_step <= 0:
            raise ValueError(
                f"Propety 't_step' should be strictly positive but a value of {new_t_step} was assigned to it.")

        self._t_step = new_t_step

    # ----

    # = `name` : `str` =

    @property
    def name(self) -> str:
        """**Property of `PWL` class**

        Summary
        -------
        Property defining the name of a `PWL` object for verbose output printing.

        Raises
        ------
        * `TypeError` : Raised if the assigned value is not a string.
        * `ValueError` : Raised if the assigned value is an empty string or already in use.
        """

        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        if not isinstance(new_name, str):
            raise TypeError(
                f"Property 'name' should be a string but an object of type '{type(new_name).__name__}' was assigned to it.")
        if not new_name:
            raise ValueError(
                "An empty string cannot be assigned to the 'name' property.")

        if new_name in PWL. __dict_of_objects:
            raise ValueError(f"Name '{new_name}' already in use.")

        PWL. __dict_of_objects.pop(self._name)
        PWL. __dict_of_objects[new_name] = self
        self._name = new_name

    # ----

    # = `verbose` : `bool` =

    @property
    def verbose(self) -> bool:
        """**Property of `PWL` class**

        Summary
        -------
        Property defining if verbose output should be printed or not.

        Raises
        ------
        * `TypeError` : Raised if the assigned value is not a boolean.
        """

        return self._verbose

    @verbose.setter
    def verbose(self, new_verbose: bool) -> None:

        if not isinstance(new_verbose, bool):
            raise TypeError(
                f"Attribute 'verbose' should be a boolean but an object of type '{type(new_verbose).__name__}' was assigned to it.")
        self._verbose = new_verbose

    # ----

    # = `hold` =

    def hold(self, duration: float) -> None:
        """**Method of `PWL` class**

        Summary
        -------
        Method that holds the last value from the previous event for a given duration.

        If the `PWL` object is empty, adds the point `(0, 0)` and holds that.

        Parameters
        ----------
        * `duration` (`float`) : Duration to hold the last value for. Should be strictly positive.

        Raises
        ------
        * `TypeError` : Raised if `duration` is not a real number.
        * `ValueError` : Raised if `duration` is not strictly positive.
        * `PrecisionError` : Raised if computational noise causes the time coordinates to not be strictly increasing.
        """

        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")

        if self._verbose:
            print(f"{self._name}: Adding hold with duration of {duration}.")

        if len(self._t_list) == len(self._x_list) == 0:
            if self._verbose:
                print("    Empty PWL object. Adding initial (0, 0) point.")
            self._add(0, 0)

        last_t = self._t_list[-1]
        last_x = self._x_list[-1]

        self._add(last_t+duration, last_x)

    # ----

    # = `square_pulse` =

    def square_pulse(self, value: float, duration: float, t_step: Optional[float] = None) -> None:
        """**Method of `PWL` class**

        Summary
        -------
        Generates a square pulse with given amplitude and duration




        """

        if t_step is None:
            t_step = self._t_step

        if not isinstance(value, Real):
            raise TypeError(
                f"Argument 'value' should be a real number but has type '{type(value).__name__}'.")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should either be a real number but has type '{type(t_step).__name__}'.")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}.")

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

    # ----

    # = `sawtooth_pulse` =

    def sawtooth_pulse(self, start: float, end: float, duration: float, t_step: Optional[float] = None) -> None:

        if t_step is None:
            t_step = self._t_step

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
                f"Argument 't_step' should either be a real number but has type '{type(t_step).__name__}'.")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}.")

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

    # ----

    # = `lin_edge` =

    def lin_edge(self, target: float, duration: float) -> None:

        if not isinstance(target, Real):
            raise TypeError(
                f"Argument 'target' should be a real number but has type '{type(target).__name__}'.")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")

        self._lin_edge(target, duration, 0)

    # ----

    # = `exp_edge` =

    def exp_edge(self, target: float, duration: float, tau: float, t_step: Optional[float] = None) -> None:

        if t_step is None:
            t_step = self._t_step

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
                f"Argument 't_step' should either be a real number but has type '{type(t_step).__name__}'.")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if tau == 0:
            raise ValueError("Argument 'tau' should be non zero.")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}.")

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

    # ----

    # = `sin_edge` =

    def sin_edge(self, target: float, duration: float, t_step: Optional[float] = None) -> None:

        if t_step is None:
            t_step = self._t_step

        if not isinstance(target, Real):
            raise TypeError(
                f"Argument 'target' should be a real number but has type '{type(target).__name__}'.")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should either be a real number but has type '{type(t_step).__name__}'.")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}.")

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

        f = _sin_edge_func(
            t1=last_t, t2=last_t+duration, f1=last_x, f2=target)

        for t in numpy.arange(last_t+t_step, last_t+duration, t_step):
            self._add(t, f(t))

        self._add(last_t+duration, target)

    # ----

    # = `smoothstep_edge` =

    def smoothstep_edge(self, target: float, duration: float, t_step: Optional[float] = None) -> None:

        if t_step is None:
            t_step = self._t_step

        if not isinstance(target, Real):
            raise TypeError(
                f"Argument 'target' should be a real number but has type '{type(target).__name__}'.")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should either be a real number but has type '{type(t_step).__name__}'.")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}.")

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

    # ----

    # = `write` =

    def write(self, filename: str, precision: int = 10) -> None:

        if not isinstance(filename, str):
            raise TypeError(
                f"Argument 'filename' should be a string but has type '{type(filename).__name__}'.")
        if not isinstance(precision, int):
            raise TypeError(
                f"Argument 'precision' should be an integer but has type '{type(precision).__name__}'.")

        if precision <= 0:
            raise ValueError(
                f"Argument 'precision' should be strictly positive but has value of {precision}.")

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
                file.write(
                    f"{ti_str}    {xi_str}\n")
                last_t = ti_str

    def _add(self, t: float, x: float) -> None:
        if len(self._t_list) >= 1 and t <= self._t_list[-1]:
            raise PrecisionError(
                f"Internal Python rounding caused the time coordinates to not be strictly increasing when adding points to {self._name}.")

        if len(self._t_list) == len(self._x_list) < 2:
            self._t_list.append(t)
            self._x_list.append(x)
        else:
            self._add_with_redundancy_check(x, t)

    def _add_with_redundancy_check(self, x: float, t: float) -> None:
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

    def _lin_edge(self, target: float, duration: float, n: int) -> None:
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


def _exp_edge_func(tau: float, t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:
    A: float = (f1*numpy.exp(t1/tau) - f2*numpy.exp(t2/tau)) / \
        (numpy.exp(t1/tau) - numpy.exp(t2/tau))
    B: float = (f1 - f2)/(numpy.exp(-t1/tau) - numpy.exp(-t2/tau))

    def f(t: float) -> float:
        result: float = A + B*numpy.exp(-t/tau)
        return result

    return f


def _sin_edge_func(t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:
    fm: float = (f1+f2)/2
    tm: float = (t1+t2)/2
    T: float = 2*(t2-t1)
    w: float = 2*numpy.pi/T
    phi: float = w*tm
    A: float = f2-fm

    def f(t: float) -> float:
        result: float = fm + A*numpy.sin(w*t - phi)
        return result

    return f


def _smoothstep_edge_func(t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:
    Am = numpy.array([[1, t1, t1**2, t1**3],
                      [1, t2, t2**2, t2**3],
                      [0, 1, 2*t1, 3*t1**2],
                      [0, 1, 2*t2, 3*t2**2]])
    Bm = numpy.array([f1, f2, 0, 0])
    A, B, C, D = numpy.linalg.solve(Am, Bm)

    def f(t: float) -> float:
        result: float = A + B*t + C*t**2 + D*t**3
        return result

    return f


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
