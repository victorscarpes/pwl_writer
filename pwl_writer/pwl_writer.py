"""
*by Victor Sabiá P. Carpes*

Tested on:

* python `3.6.6`
        * numpy `1.19.5`
        * matplotlib `3.3.4`
* python `3.11.1`
        * numpy `1.25.0`
        * matplotlib `3.7.1`

Type stubs for older numpy versions for mypy checking can be found [here](https://github.com/numpy/numpy-stubs).

----

# Index

* [Package Summary](#package-summary)
* [Precision Related Exception](#precision-related-exception)
* [PWL Class](#pwl-class)
        * Dunder Methods
            * [Initializer](#initializer)
            * [String Representation](#string-representation)
            * [Length Calculator](#length-calculator)
            * [PWL Object as a Callable](#pwl-object-as-a-callable)
            * [PWL Object Slicing](#pwl-object-slicing)
            * [PWL Iterator](#pwl-iterator)
            * [PWL Multiplication](#pwl-multiplication)
        * Properties
            * [Time Coordinates](#time-coordinates)
            * [Dependent Coordinates](#dependent-coordinates)
            * [Default Timestep](#default-timestep)
            * [Name](#name)
            * [Verbose Flag](#verbose-flag)
            * [Plot Enable Flag](#plot-enable-flag) *(optional feature: requires matplotlib)*
        * Instance Methods
            * [Last Value Holder](#last-value-holder)
            * [Linear Transition](#linear-transition)
            * [Rectangular Pulse](#rectangular-pulse)
            * [Sawtooth Pulse](#sawtooth-pulse)
            * [Exponential Transition](#exponential-transition)
            * [Half Sine Transition](#half-sine-transition)
            * [Smoothstep Transition](#smoothstep-transition)
            * [PWL File Writer](#pwl-file-writer)
            * [PWL Copy](#pwl-copy)
        * Class Methods
            * [PWL Plotter](#pwl-plotter) *(optional feature: requires matplotlib)*
        * Private Methods *(minimal documentation)*
            * [PWL Point Adder](#pwl-point-adder)
            * [Colinear Points Eliminator](#colinear-points-eliminator)
            * [Nested Linear Transition](#nested-linear-transition)
* Private Functions *(minimal documentation)*
        * [Exponential Function Generator](#exponential-function-generator)
        * [Sinusoidal Function Generator](#sinusoidal-function-generator)
        * [Smoothstep Function Generator](#smoothstep-function-generator)
        
----

# Package Summary

This package defines a class `PWL` to generate objects that represent time dependent signals `x(t)` that need to be coded in a PWL file. Those objects are built using little components such as rectangular pulses and sawtooth pulses that can be chained together.

The motivations for this package are the nuisances of writing PWL files by hand. To properly explain this, let's discuss how PWL files work.

PWL stands for piecewise linear. A PWL file is a way to represent a time dependent signal (referenced by `x(t)` from now on) for simulation softwares such as LTspice and Cadence Virtuoso. In it, points of the form `(t, x)` are stored. During simulation, those points are interpolated with first degree polynomials. This poses 2 problems:

1. Due to the linear interpolation, the resulting signal is continuous. This tends to be desirable, but if the intention is moddeling, for example, rectangular pulses, each transition will need 2 points with very close time coordinates to approximate a discontinuous transition. This can get extremely tedious to code out by hand.

2. Each point has an absolute time coordinate with respect to the origin. If the desired signal is for example a series of rectangular pulses with certain durations and for some reason the duration of the first pulse is changed, all the following points will need to have their time coordinates changed as well.

This package solves both problems by providing an abstraction layer. They are solved by the 2 following features:

1. A minimal timestep is defined at the creation of the PWL object that is used to automatically generate all the needed transitions for any discontinous transition.

2. The signal is built using small building blocks (such as rectangular pulse and exponential transition) called events that are defined in terms of durations. That is to say, time is treated in a differential fashion. The time coordinates from a given event are all with respect to the final instant of the previous event. For example, let's assume we want to model a rectangular pulse with amplitude 1 and duration 1 second followed by a downtime at zero for 10 seconds and then another rectangular pulse with the same duration and amplitude. If we change the duration of the first pulse to 2 seconds, the downtime and second pulse will be both  delayed by the 1 second but retain their durations.

Another advantage of using this package is not a feature per se but more a consequence of using a programing language. That advantage is simply that all those events can be added inside for loops, if clauses and functions, allowing for greater flexibility. For example, let's assume we want to control a system that can be in the following states:

* Idle
* Mode 1
* Mode 2

For each state, various control signals need to be at specific values. We could create a `PWL` object for each control signal and define 3 functions that apply all the needed values for the control signals for each state. If we nedded the system to be at mode 1 for 3 seconds, idle for 1 second and at mode 2 for 5 seconds, we could write something like the following:

        mode1_state(3)
        idle_state(1)
        mode2_state(5)
"""

__all__ = ['PrecisionError', 'PWL']

from warnings import warn
from numbers import Real
from typing import Callable, List, Optional, TYPE_CHECKING, Union, Tuple, Iterator
import weakref
import numpy as np


try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:
    _has_matplotlib = False
    warn("Matplotlib package not found. Optional plotting features deactivated.", ImportWarning)
else:
    _has_matplotlib = True

if TYPE_CHECKING:
    WeakDict = weakref.WeakValueDictionary[str, "PWL"]

# ----

# == Precision Related Exception ==


class PrecisionError(Exception):
    """**`PrecisionError` exception class**

    This class defines an exception meant to be raised when any type of rounding or loss of precision that causes the time coordinates of a PWL object to not be strictly increasing.
    """


# ----

# == PWL Class ==

class PWL():
    """**`PWL` class**

    This class defines an object that represnts a time dependent signal `x(t)`. Those objects can operated on by methods to build, event by event, the desired signal as described on the package introduction.
    """

    __dict_of_objects = weakref.WeakValueDictionary()  # type: WeakDict

    # ----

    # == Initializer ==

    def __init__(self, t_step: float, name: Optional[str] = None, verbose: bool = False) -> None:
        """**`__init__` dunder method of `PWL` class**

        ### Summary

        Initializer for the `PWL` class.

        ### Arguments

        * `t_step` (`float`) : Default timestep for all operations. Should be strictly positive.
        * `name` (`str`, optional) : Name of the `PWL` object used for verbose output printing. Should not be empty. If not set, automatically generates a name based on already taken names.
        * `verbose` (`bool`, optional) : Flag indicating if verbose output should be printed. If not set, defaults to `False`.

        ### Raises

        * `TypeError` : Raised if either `t_step` is not a real number, `name` is not a string or `verbose` is not a boolean.
        * `ValueError` : Raised if `t_step` is not strictly positive or `name` is either empty or already taken.
        """

        if name is None:
            i: int = 0
            while f"pwl_{i}" in PWL.__dict_of_objects.keys():
                i += 1
            name = f"pwl_{i}"

        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should be a real number but has type '{type(t_step).__name__}'")
        if not isinstance(name, str):
            raise TypeError(
                f"Argument 'name' should either be a string but has type '{type(name).__name__}'")
        if not isinstance(verbose, bool):
            raise TypeError(
                f"Argument 'verbose' should be a boolean but has type '{type(verbose).__name__}'")

        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}")
        if not name:
            raise ValueError("Argument 'name' should not be empty")

        self._t_list: List[float] = []
        self._x_list: List[float] = []
        self._t_step: float = t_step
        self._name: str = name
        self._verbose: bool = verbose
        self._plot_flag: bool = True

        if name in PWL.__dict_of_objects.keys():
            raise ValueError(f"Name '{name}' already in use")

        PWL.__dict_of_objects[name] = self

    # ----

    # == String Representation ==

    def __str__(self) -> str:
        """**`__str__` dunder method of `PWL` class**

        ### Summary

        String representation of `PWL` instances in the form `[name]: PWL object with [# of points] and duration of [total time duration] seconds`.

        ### Returns

        * `str`
        """

        duration = 0 if len(self._t_list) == 0 else max(self._t_list)

        return f"{self.name}: PWL object with {len(self._t_list)} points and duration of {duration} seconds"

    # ----

    # == Length Calculator ==

    def __len__(self) -> int:
        """**`__len__` dunder method of `PWL` class**

        ### Summary

        Length of `PWL` instances defined as the number of `(t, x)` points they contain.

        ### Returns

        * `int`
        """

        return len(self._t_list)

    # ----

    # == PWL Object as a Callable ==

    def __call__(self, t: float) -> float:
        """**`__call__` dunder method of `PWL` class**

        ### Summary

        Call `PWL` object as a function by linearly interpolating between it's time and dependent coordinates.

        ### Arguments

        * `t` (`float`) : Time instant to evaluate the object at. If negative, returns zero. If bigger than the duration of the object, returns the object's last dependent coordinate.

        ### Returns

        * `float`

        ### Raises

        * `TypeError` : Raised if `t` is not a real number.
        """

        if not isinstance(t, Real):
            raise TypeError(
                f"Argument 't' should be a real number but has type '{type(t).__name__}'")

        t_list = self._t_list
        x_list = self._x_list

        return np.interp(x=t, xp=t_list, fp=x_list, left=0)

    # ----

    # == PWL Object Slicing ==

    def __getitem__(self, index: Union[int, slice]) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """**`__getitem__` dunder method of `PWL` class**

        ### Summary

        Slice `PWL` objects.

        ### Arguments

        * `index` (`int` or `slice`) : Index for list of `(t, x)` points.

        ### Returns

        * `Tuple[float, float]` : Returned if single index is passed.
        * `List[Tuple[float, float]]` : Rerturned if multiple indices are passed.

        ### Raises

        * `TypeError` : Raised if `index` is not an integer or slice.
        * `IndexError` : Raised if `index` is out of bounds.
        """

        if not isinstance(index, (int, slice)):
            raise TypeError(
                f"PWL indices must be integers or slices, not {type(index).__name__}")

        coordinates_pair = list(zip(self.t_list, self.x_list))

        return coordinates_pair[index]

    # ----

    # == PWL Iterator ==

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        """**`__iter__` dunder method of `PWL` class**

        ### Summary

        Creates a generator object that yields all the `(t, x)` points as tuples.

        ### Yields

        * `Tuple[float, float]`
        """
        yield from list(zip(self.t_list, self.x_list))

    # ----

    # == PWL Multiplication ==

    def __mul__(self, other: Union["PWL", float]) -> "PWL":
        """**`__mul__` and `__rmul__`  dunder methods of `PWL` class**

        ### Summary

        Implements point-wise multiplication of `PWL` objects with real numbers and other `PWL` objects.

        The new `PWL` objects created has `t_step` equal to the lower `t_step` between the operands.

        If one operand is longer than the other, extends the sorter one by holding it's last value.

        ### Arguments

        * `other` (`PWL` or `float`) : Thing to multiply the `PWL` object by.

        ### Returns

        * `PWL`

        ### Raises

        * `TypeError` : Raised if operation is not implemented between the operands.
        """

        if not isinstance(other, (Real, PWL)):
            return NotImplemented

        t_step = min(self.t_step, other.t_step) if isinstance(
            other, PWL) else self.t_step
        new_pwl = PWL(t_step=t_step)

        if isinstance(other, Real):
            for t, x in self:
                new_pwl._add(t, float(other)*x)

        else:
            unsorted_t_set = set(self.t_list + other.t_list)
            t_list = sorted(list(unsorted_t_set))
            for t in t_list:
                new_pwl._add(t, self(t) * other(t))

        return new_pwl

    __rmul__ = __mul__

    # ----

    # == Time Coordinates ==

    @property
    def t_list(self) -> List[float]:
        """**`t_list` property of `PWL` class**

        ### Type

        * `list[float]`

        ### Summary

        Read only property containing all the time coordinates of a `PWL` object.

        ### Raises

        * `AttributeError` : Raised if assignment was attempetd.
        """

        return self._t_list[:]

    # ----

    # == Dependent Coordinates ==

    @property
    def x_list(self) -> List[float]:
        """**`x_list` property of `PWL` class**

        ### Type

        * `list[float]`

        ### Summary

        Read only property containing all the dependent coordinates of a `PWL` object.

        ### Raises

        * `AttributeError` : Raised if assignment was attempetd.
        """

        return self._x_list[:]

    # ----

    # == Default Timestep ==

    @property
    def t_step(self) -> float:
        """**`t_step` property of `PWL` class**

        ### Type

        * `float`

        ### Summary

        Property defining the default timestep of a `PWL` object.

        ### Raises

        * `TypeError` : Raised if the assigned value is not a real number.
        * `ValueError` : Raised if the assigned value is not strictly positive.
        """

        return self._t_step

    @t_step.setter
    def t_step(self, new_t_step: float) -> None:
        if not isinstance(new_t_step, float):
            raise TypeError(
                f"Property 't_step' should be a real number but an object of type '{type(new_t_step).__name__}' was assigned to it")
        if new_t_step <= 0:
            raise ValueError(
                f"Propety 't_step' should be strictly positive but a value of {new_t_step} was assigned to it")

        self._t_step = new_t_step

    # ----

    # == Name ==

    @property
    def name(self) -> str:
        """**`name` property of `PWL` class**

        ### Type

        * `str`

        ### Summary

        Property defining the name of a `PWL` object for verbose output printing.

        ### Raises

        * `TypeError` : Raised if the assigned value is not a string.
        * `ValueError` : Raised if the assigned value is an empty string or already in use.
        """

        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        if not isinstance(new_name, str):
            raise TypeError(
                f"Property 'name' should be a string but an object of type '{type(new_name).__name__}' was assigned to it")
        if not new_name:
            raise ValueError(
                "An empty string cannot be assigned to the 'name' property")

        if new_name in PWL.__dict_of_objects.keys():
            raise ValueError(f"Name '{new_name}' already in use")

        PWL.__dict_of_objects.pop(self._name)
        PWL.__dict_of_objects[new_name] = self
        self._name = new_name

    # ----

    # == Verbose Flag ==

    @property
    def verbose(self) -> bool:
        """**`verbose` property of `PWL` class**

        ### Type

        * `bool`

        ### Summary

        Property defining if verbose output should be printed or not.

        ### Raises

        * `TypeError` : Raised if the assigned value is not a boolean.
        """

        return self._verbose

    @verbose.setter
    def verbose(self, new_verbose: bool) -> None:

        if not isinstance(new_verbose, bool):
            raise TypeError(
                f"Attribute 'verbose' should be a boolean but an object of type '{type(new_verbose).__name__}' was assigned to it")
        self._verbose = new_verbose

    # ----

    # == Plot Enable Flag ==

    @property
    def plot_flag(self) -> bool:
        """**`plot_flag` property of `PWL` class**
        *Optional feature: Requires matplotlib*

        ### Type

        * `bool`

        ### Summary

        Property defining if object should be plotted by the PWL plotter method.

        ### Raises

        * `TypeError` : Raised if the assigned value is not a boolean.

        ### See Also

        * [PWL Plotter](#pwl-plotter)
        """

        if (not _has_matplotlib) and self._verbose:
            print(
                "Optional features deactivated. Using the plot_flag does nothing in this case")

        return self._plot_flag

    @plot_flag.setter
    def plot_flag(self, new_plot_flag: bool) -> None:

        if not isinstance(new_plot_flag, bool):
            raise TypeError(
                f"Attribute 'plot_flag' should be a boolean but an object of type '{type(new_plot_flag).__name__}' was assigned to it")

        if (not _has_matplotlib) and self._verbose:
            print(
                "Optional features deactivated. Using the plot_flag does nothing in this case")

        self._plot_flag = new_plot_flag

    # ----

    # == Last Value Holder ==

    def hold(self, duration: float) -> None:
        """**`hold` method of `PWL` class**

        ### Summary

        Method that holds the last value from the previous event for a given duration.

        If the `PWL` object is empty, adds the point `(0, 0)` and holds that.

        ### Parameters

        * `duration` (`float`) : Duration to hold the last value for. Should be strictly positive.

        ### Raises

        * `TypeError` : Raised if `duration` is not a real number.
        * `ValueError` : Raised if `duration` is not strictly positive.
        * `PrecisionError` : Raised if computational noise causes the time coordinates to not be strictly increasing.
        """

        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}")

        if self._verbose:
            print(f"{self._name}: Adding hold with duration of {duration}")

        if len(self._t_list) == len(self._x_list) == 0:
            if self._verbose:
                print("    Empty PWL object. Adding initial (0, 0) point")
            self._add(0, 0)

        last_t = self._t_list[-1]
        last_x = self._x_list[-1]

        self._add(last_t+duration, last_x)

    # ----

    # == Linear Transition ==

    def lin_transition(self, target: float, duration: float) -> None:
        """**`lin_transition` method of `PWL` class**

        ### Summary

        Method that generates a linear transition from the last value of the previous event to a given target with a given duration.

        If the `PWL` object is empty, adds the point `(0, 0)` and transitions from that.

        ### Arguments

        * `target` (`float`) : Value to transition to.
        * `duration` (`float`) : Duration of the transition. Should be strictly positive.

        ### Raises

        * `TypeError` : Raised if either `target` or duration` is not a real number.
        * `ValueError` : Raised if `duration` is not strictly positive.
        * `PrecisionError` : Raised if computational noise causes the time coordinates to not be strictly increasing.
        """

        if not isinstance(target, Real):
            raise TypeError(
                f"Argument 'target' should be a real number but has type '{type(target).__name__}'")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}")

        self._lin_transition(target, duration, 0)

    # ----

    # == Rectangular Pulse ==

    def rect_pulse(self, value: float, duration: float, t_step: Optional[float] = None) -> None:
        """**`rect_pulse` method of `PWL` class**

        ### Summary

        Method that generates a rectangular pulse with given amplitude and duration.

        If `duration` is less than or equal to `t_step` (`self.t_step` if `t_step` is not set), substitutes the pulse by a linear transition from the last value of the previous event to `value` with duration `t_step` (`self.t_step` if `t_step` is not set).

        ### Arguments

        * `value` (`float`) : Amplitude of the pulse.
        * `duration` (`float`) : Duration of the pulse. Should be strictly positive.
        * `t_step` (`float`, optional) : Transition time for the discontinuity. Should be strictly positive. If not set, uses `self.t_step`.

        ### Raises

        * `TypeError` : Raised if either `value`, duration` or `t_step` is not a real number.
        * `ValueError` : Raised if either `duration` or `t_step` is not strictly positive.
        * `PrecisionError` : Raised if computational noise causes the time coordinates to not be strictly increasing.

        ### See Also

        * [Linear Transition](#linear-transition)
        """

        if t_step is None:
            t_step = self._t_step

        if not isinstance(value, Real):
            raise TypeError(
                f"Argument 'value' should be a real number but has type '{type(value).__name__}'")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'")
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should either be a real number but has type '{type(t_step).__name__}'")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}")

        if self._verbose:
            print(f"{self._name}: Adding rectangular pulse with value of {value}, duration of {duration} and time step of {t_step}")

        if duration <= t_step:
            if self._verbose:
                print(
                    f"{self._name}: Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear transition")
            self._lin_transition(value, t_step, 1)
            return

        if len(self._t_list) == len(self._x_list) == 0:
            self._add(0, value)
            last_t = 0
        else:
            last_t = self._t_list[-1]
            self._add(last_t+t_step, value)

        self._add(last_t+duration, value)

    # ----

    # == Sawtooth Pulse ==

    def sawtooth_pulse(self, start: float, end: float, duration: float, t_step: Optional[float] = None) -> None:
        """**`sawtooth_pulse` method of `PWL` class**

        ### Summary

        Method that generates a sawtooth pulse with given starting and ending amplitudes and duration.

        If `duration` is less than or equal to `t_step` (`self.t_step` if `t_step` is not set), substitutes the pulse by a linear transition from the last value of the previous event to `end` with duration `t_step` (`self.t_step` if `t_step` is not set).

        ### Arguments

        * `start` (`float`) : Amplitude at the start of the pulse.
        * `end` (`float`) : Amplitude at the end of the pulse.
        * `duration` (`float`) : Duration of the pulse. Should be strictly positive.
        * `t_step` (`float`, optional) : Transition time for the discontinuity. Should be strictly positive. If not set, uses `self.t_step`.

        ### Raises

        * `TypeError` : Raised if either `start`, `end`, duration` or `t_step` is not a real number.
        * `ValueError` : Raised if either `duration` or `t_step` is not strictly positive.
        * `PrecisionError` : Raised if computational noise causes the time coordinates to not be strictly increasing.

        ### See Also

        * [Linear Transition](#linear-transition)
        """

        if t_step is None:
            t_step = self._t_step

        if not isinstance(start, Real):
            raise TypeError(
                f"Argument 'start' should be a real number but has type '{type(start).__name__}'")
        if not isinstance(end, Real):
            raise TypeError(
                f"Argument 'end' should be a real number but has type '{type(end).__name__}'")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'")
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should either be a real number but has type '{type(t_step).__name__}'")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}")

        if self._verbose:
            print(f"{self._name}: Adding sawtoth pulse from {start} to {end} with duration of {duration} and time step of {t_step}")

        if duration <= t_step:
            if self._verbose:
                print(
                    f"{self._name}: Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear transition")
            self._lin_transition(end, t_step, 1)
            return

        if len(self._t_list) == len(self._x_list) == 0:
            self._add(0, start)
            last_t = 0
        else:
            last_t = self._t_list[-1]
            self._add(last_t+t_step, start)

        self._add(last_t+duration, end)

    # ----

    # == Exponential Transition ==

    def exp_transition(self, target: float, duration: float, tau: float, t_step: Optional[float] = None) -> None:
        """**`exp_transition` method of `PWL` class**

        ### Summary

        Method that generates an exponential transition from the last value of the previous event to a given target with a given duration.

        If the `PWL` object is empty, adds the point `(0, 0)` and transitions from that.

        Let's call `x0` the last value of the previous event and `t0` the instant when the transition begins. The transition will follow thw following form:

            f(t) = A + B*exp(-t/tau)

        The constants `A` and `B` are chosen such that the following conditions are met:

            f(t0) = x0
            f(t0 + duration) = target

        The sign of `tau` defines if `f(t)` diverges or converges when `t` goes to positive infinity.

        If `duration` is less than or equal to `t_step` (`self.t_step` if `t_step` is not set), substitutes the pulse by a linear transition from the last value of the previous event to `target` with duration `t_step` (`self.t_step` if `t_step` is not set).

        ### Arguments

        * `target` (`float`) : Value to transition to.
        * `duration` (`float`) : Duration of the transition. Should be strictly positive.
        * `tau` (`float`) : Time constant of the exponential. SHould be non zero.
        * `t_step` (`float`, optional) : Timestep between consecutive points inside the transition. Should be strictly positive. If not set, uses `self.t_step`.

        ### Raises

        * `TypeError` : Raised if either `target`, `duration`, tau` or `t_step` is not a real number.
        * `ValueError` : Raised if either `duration` or `t_step` is not strictly positive or `tau` is equal to zero.
        * `PrecisionError` : Raised if computational noise causes the time coordinates to not be strictly increasing.

        ### See Also

        * [Linear Transition](#linear-transition)
        """

        if t_step is None:
            t_step = self._t_step

        if not isinstance(target, Real):
            raise TypeError(
                f"Argument 'target' should be a real number but has type '{type(target).__name__}'")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'")
        if not isinstance(tau, Real):
            raise TypeError(
                f"Argument 'tau' should be a real number but has type '{type(tau).__name__}'")
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should either be a real number but has type '{type(t_step).__name__}'")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}")
        if tau == 0:
            raise ValueError("Argument 'tau' should be non zero")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}")

        if self._verbose:
            print(f"{self._name}: Adding exponential transition with target of {target}, time constant of {tau}, duration of {duration} and time step of {t_step}")

        if duration <= t_step:
            if self._verbose:
                print(
                    f"    Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear transition")
            self._lin_transition(target, t_step, 2)
            return

        if len(self._t_list) == len(self._x_list) == 0:
            if self._verbose:
                print("    Empty PWL object. Adding initial (0, 0) point")
            self._add(0, 0)

        last_t = self._t_list[-1]
        last_x = self._x_list[-1]

        f = _exp_transition_func(tau=tau, t1=last_t, t2=last_t +
                                 duration, f1=last_x, f2=target)

        for t in np.arange(last_t+t_step, last_t+duration, t_step):
            self._add(t, f(t))

        self._add(last_t+duration, target)

    # ----

    # == Half Sine Transition ==

    def sin_transition(self, target: float, duration: float, t_step: Optional[float] = None) -> None:
        """**`sin_transition` method of `PWL` class**

        ### Summary

        Method that generates a half sine transition from the last value of the previous event to a given target with a given duration.

        If the `PWL` object is empty, adds the point `(0, 0)` and transitions from that.

        Let's call `x0` the last value of the previous event and `t0` the instant when the transition begins. The transition will follow thw following form:

            f(t) = A + B*sin(w*t - phi)

        The constants `A`, `B`, `w` and `phi` are chosen shuch that the following conditions are met:

            f(t0) = x0
            f(t0 + duration) = target
            f'(t0) = f'(t0 + duration) = 0

        Due to the periodic nature of sine, inifinite solutions for `f(t)` that satisfy those conditions exist. The only monotonic solution is chosen. That is to say, the wavelength of the chopse solution is equal to `2*duration`.

        If `duration` is less than or equal to `t_step` (`self.t_step` if `t_step` is not set), substitutes the pulse by a linear transition from the last value of the previous event to `target` with duration `t_step` (`self.t_step` if `t_step` is not set).

        ### Arguments

        * `target` (`float`) : Value to transition to.
        * `duration` (`float`) : Duration of the transition. Should be strictly positive.
        * `t_step` (`float`, optional) : Timestep between consecutive points inside the transition. Should be strictly positive. If not set, uses `self.t_step`.

        ### Raises

        * `TypeError` : Raised if either `target`, `duration` or `t_step` is not a real number.
        * `ValueError` : Raised if either `duration` or `t_step` is not strictly positive.
        * `PrecisionError` : Raised if computational noise causes the time coordinates to not be strictly increasing.

        ### See Also

        * [Linear Transition](#linear-transition)
        """

        if t_step is None:
            t_step = self._t_step

        if not isinstance(target, Real):
            raise TypeError(
                f"Argument 'target' should be a real number but has type '{type(target).__name__}'")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'")
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should either be a real number but has type '{type(t_step).__name__}'")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}")

        if self._verbose:
            print(f"{self._name}: Adding sinusoidal transition with target of {target}, duration of {duration} and time step of {t_step}")

        if duration <= t_step:
            if self._verbose:
                print(
                    f"    Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear transition")
            self._lin_transition(target, t_step, n=2)
            return

        if len(self._t_list) == len(self._x_list) == 0:
            if self._verbose:
                print("    Empty PWL object. Adding initial (0, 0) point")
            self._add(0, 0)

        last_t = self._t_list[-1]
        last_x = self._x_list[-1]

        f = _sin_transition_func(
            t1=last_t, t2=last_t+duration, f1=last_x, f2=target)

        for t in np.arange(last_t+t_step, last_t+duration, t_step):
            self._add(t, f(t))

        self._add(last_t+duration, target)

    # ----

    # == Smoothstep Transition ==

    def smoothstep_transition(self, target: float, duration: float, t_step: Optional[float] = None) -> None:
        """**`smoothstep_transition` method of `PWL` class**

        ### Summary

        Method that generates a smoothstep transition from the last value of the previous event to a given target with a given duration.

        If the `PWL` object is empty, adds the point `(0, 0)` and transitions from that.

        Let's call `x0` the last value of the previous event and `t0` the instant when the transition begins. The transition will follow thw following form:

            f(t) = A + B*t + C*t^2 + D*t^3

        The constants `A`, `B`, `C` and `D` are chosen shuch that the following conditions are met:

            f(t0) = x0
            f(t0 + duration) = target
            f'(t0) = f'(t0 + duration) = 0

        If `duration` is less than or equal to `t_step` (`self.t_step` if `t_step` is not set), substitutes the pulse by a linear transition from the last value of the previous event to `target` with duration `t_step` (`self.t_step` if `t_step` is not set).

        ### Arguments

        * `target` (`float`) : Value to transition to.
        * `duration` (`float`) : Duration of the transition. Should be strictly positive.
        * `t_step` (`float`, optional) : Timestep between consecutive points inside the transition. Should be strictly positive. If not set, uses `self.t_step`.

        ### Raises

        * `TypeError` : Raised if either `target`, `duration` or `t_step` is not a real number.
        * `ValueError` : Raised if either `duration` or `t_step` is not strictly positive.
        * `PrecisionError` : Raised if computational noise causes the time coordinates to not be strictly increasing.

        ### See Also

        * [Linear Transition](#linear-transition)
        """

        if t_step is None:
            t_step = self._t_step

        if not isinstance(target, Real):
            raise TypeError(
                f"Argument 'target' should be a real number but has type '{type(target).__name__}'")
        if not isinstance(duration, Real):
            raise TypeError(
                f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'")
        if not isinstance(t_step, Real):
            raise TypeError(
                f"Argument 't_step' should either be a real number but has type '{type(t_step).__name__}'")

        if duration <= 0:
            raise ValueError(
                f"Argument 'duration' should be strictly positive but has value of {duration}")
        if t_step <= 0:
            raise ValueError(
                f"Argument 't_step' should be strictly positive but has value of {t_step}")

        if self._verbose:
            print(f"{self._name}: Adding smoothstep transition with target of {target}, duration of {duration} and time step of {t_step}")

        if duration <= t_step:
            if self._verbose:
                print(
                    f"    Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear transition")
            self._lin_transition(target, t_step, n=2)
            return

        if len(self._t_list) == len(self._x_list) == 0:
            if self._verbose:
                print("    Empty PWL object. Adding initial (0, 0) point")
            self._add(0, 0)

        last_t = self._t_list[-1]
        last_x = self._x_list[-1]

        f = _smoothstep_transition_func(
            t1=last_t, t2=last_t+duration, f1=last_x, f2=target)

        for t in np.arange(last_t+t_step, last_t+duration, t_step):
            self._add(t, f(t))

        self._add(last_t+duration, target)

    # ----

    # == PWL File Writer ==

    def write(self, filename: Optional[str] = None, precision: int = 10) -> None:
        """**`write` method of `PWL` class**

        ### Summary

        Method that takes a `PWL` object and writes a PWL file with it's `(t, x)` coordinates in scientific notation.

        If the specified file already exists, overwrites it.

        ### Arguments

        * `filename` (`str`, optional) : Name of file to be created. If not set, uses `self.name` if an added `.txt` extension.
        * `precision` (`int`, optional) : Number of significant figures used when writing the PWL file. Should be strictly positive. If not set, defaults to 10.

        Raises
        ------
        * `TypeError` : Raised if `filename` is not a string or `precision` is not an integer.
        * `ValueError` : Raised if `precision` is not strictly positive.
        * `PrecisionError` : Raised if `precision` is such that the rounding causes the time coordinates to not be strictly increasing.
        """

        if filename is None:
            filename = f'{self._name}.txt'

        if not isinstance(filename, str):
            raise TypeError(
                f"Argument 'filename' should be a string but has type '{type(filename).__name__}'")
        if not isinstance(precision, int):
            raise TypeError(
                f"Argument 'precision' should be an integer but has type '{type(precision).__name__}'")

        if precision <= 0:
            raise ValueError(
                f"Argument 'precision' should be strictly positive but has value of {precision}")

        if self._verbose:
            print(f"{self._name}: Writing PWL file to {filename}")

        t_list = self._t_list
        x_list = self._x_list

        with open(filename, "w") as file:
            ti_str = np.format_float_scientific(
                t_list[0], precision-1, unique=False, sign=False)
            xi_str = np.format_float_scientific(
                x_list[0], precision-1, unique=False, sign=True)
            file.write(f"{ti_str}    {xi_str}\n")
            last_t = ti_str
            for ti, xi in zip(t_list[1:], x_list[1:]):
                ti_str = np.format_float_scientific(
                    ti, precision-1, unique=False, sign=False)
                xi_str = np.format_float_scientific(
                    xi, precision-1, unique=False, sign=True)
                if ti_str == last_t:
                    raise PrecisionError(
                        "The chosen precision level caused the written time coordinates to not be strictly increasing")
                file.write(
                    f"{ti_str}    {xi_str}\n")
                last_t = ti_str

    # ----

    # == PWL Copy ==

    def copy(self, name: Optional[str] = None) -> "PWL":
        """**`copy` class method of `PWL` class**

        ### Summary

        Method that creates a deep copy of a `PWL` object.



        ### Arguments

        * `name` (`str`, optional) : Name of the `PWL` object used for verbose output printing. Should not be empty. If not set, automatically generates a name based on already taken names. 

        ### Returns

        * `PWL`

        ### Raises

        * `TypeError` : Raised if `name` is not a string.
        * `ValueError` : Raised if `name` is either empty or already taken.
        """

        new_pwl = PWL(t_step=self.t_step, name=name, verbose=self.verbose)

        new_pwl._t_list = self.t_list
        new_pwl._x_list = self.x_list

        return new_pwl

    # ----

    # == PWL Plotter ==

    @classmethod
    def plot(cls, merge: bool = False) -> None:
        """**`plot` class method of `PWL` class**
        *Optional feature: Requires matplotlib*

        ### Summary

        Class method that takes all instances of the `PWL` class with plot enable flag set to `True` and plots them on the same time axis.

        ### Arguments

        * `merge` (`bool`, optional) : Flag indicating if all signals should be ploted on the same strip or separeted. If not set, defaults to False.

        ### Raises

        * `TypeError` : Raised if `merge` is not a boolean.
        * `ImportError` : Raised if the matplotlib package is not installed.

        ### See Also

        * [Plot Enable Flag](#plot-enable-flag)
        """

        if not _has_matplotlib:
            raise ImportError(
                "Optional plotting features are deactivated. Install matplotlib to use")

        if not isinstance(merge, bool):
            raise TypeError(
                f"Argument 'merge' should be a boolean but has type '{type(merge).__name__}'")

        dict_of_objects = {key: pwl for key,
                           pwl in cls.__dict_of_objects.items() if pwl.plot_flag}

        if not dict_of_objects:
            return None

        if merge:
            axs = plt.subplots(nrows=1, sharex=True, squeeze=False)[1]
            axs = np.repeat(axs, len(dict_of_objects))
        else:
            axs = plt.subplots(nrows=len(dict_of_objects),
                               sharex=True, squeeze=False)[1]
            axs = axs.flatten()
        x_max: float = 0

        for key, ax in zip(dict_of_objects, axs):
            pwl = dict_of_objects[key]
            if not pwl._plot_flag:
                continue
            x_list = pwl.t_list
            x_max = max(x_max, max(x_list))
            y_list = pwl.x_list
            label = pwl.name
            ax.plot(x_list, y_list)
            ax.set_ylabel(label)

        axs[0].set_xlim(xmin=0, xmax=x_max)
        plt.show()

    # = Private Methods and Functions =

    # From this point forward, all listed methods and functions are not intended to be used by the user. Brief descriptions will be provided, but documentation will be kept to a minimum.

    # ----

    # == PWL Point Adder ==

    def _add(self, t: float, x: float) -> None:
        """**`_add` private method of `PWL` class**

        Private method that adds a `(t, x)` point to a `PWL` object of any size.
        """

        if len(self._t_list) >= 1 and t <= self._t_list[-1]:
            raise PrecisionError(
                f"Internal Python rounding caused the time coordinates to not be strictly increasing when adding points to {self._name}")

        if len(self._t_list) == len(self._x_list) < 2:
            self._t_list.append(t)
            self._x_list.append(x)
        else:
            self._colinear_eliminator(x, t)

    # ----

    # == Colinear Points Eliminator ==

    def _colinear_eliminator(self, x: float, t: float) -> None:
        """**`_colinear_eliminator` private method of `PWL` class**

        Private method that adds a `(t, x)` point to a `PWL` object with 2 or more points without consecutive colinear points.
        """

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

    # ----

    # == Nested Linear Transition ==

    def _lin_transition(self, target: float, duration: float, n: int) -> None:
        """**`_lin_transition` private method of `PWL` class**

        Private method to generate a linear transition. The difference between this method and the [public linear transition](#linear-transition) method is that this method prints indented verbose output when called from within any of the public methods that revert to a linear transition in certain conditions.
        """

        if self._verbose:
            if n == 0:
                print(
                    f"{self._name}: Adding linear transition with target of {target} and duration of {duration}")
            else:
                print(
                    n*"    "+f"Adding linear transition with target of {target} and duration of {duration}")

        if len(self._t_list) == len(self._x_list) == 0:
            if self._verbose:
                print((n+1)*"    "+"Empty PWL object. Adding initial (0, 0) point")
            self._add(0, 0)

        last_t = self._t_list[-1]
        self._add(last_t+duration, target)

# ----

# == Exponential Function Generator ==


def _exp_transition_func(tau: float, t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:
    """**`_exp_transition_func` private function**

    Private function that generates an exponential function passing trough 2 fixed points.
    """

    A: float = (f1*np.exp(t1/tau) - f2*np.exp(t2/tau)) / \
        (np.exp(t1/tau) - np.exp(t2/tau))
    B: float = (f1 - f2)/(np.exp(-t1/tau) - np.exp(-t2/tau))

    def f(t: float) -> float:
        result: float = A + B*np.exp(-t/tau)
        return result

    return f

# ----

# == Sinusoidal Function Generator ==


def _sin_transition_func(t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:
    """**`_sin_transition_func` private function**

    Private function that generates a sinusoidal function passing trough 2 fixed points.
    """

    fm: float = (f1+f2)/2
    tm: float = (t1+t2)/2
    T: float = 2*(t2-t1)
    w: float = 2*np.pi/T
    phi: float = w*tm
    A: float = f2-fm

    def f(t: float) -> float:
        result: float = fm + A*np.sin(w*t - phi)
        return result

    return f

# ----

# == Smoothstep Function Generator ==


def _smoothstep_transition_func(t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:
    """**`_smoothstep_transition_func` private function**

    Private function that generates a smoothstep function passing trough 2 fixed points.
    """

    Am = np.array([[1, t1, t1**2, t1**3],
                   [1, t2, t2**2, t2**3],
                   [0, 1, 2*t1, 3*t1**2],
                   [0, 1, 2*t2, 3*t2**2]])
    Bm = np.array([f1, f2, 0, 0])
    A, B, C, D = np.linalg.solve(Am, Bm)

    def f(t: float) -> float:
        result: float = A + B*t + C*t**2 + D*t**3
        return result

    return f

# ----


if __name__ == "__main__":
    pwl0 = PWL(0.001)
    pwl0.hold(1)
    pwl0.sin_transition(1, 1)
    pwl0.hold(1)
    pwl0.sin_transition(-1, 1)
    pwl0.hold(1)
    pwl0.sin_transition(1, 1)

    PWL.plot(merge=False)
