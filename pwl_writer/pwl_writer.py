"""
This module defines a class `PWL` to generate objects 
"""


from numbers import Real
import numpy as np  # type: ignore
from typing import Optional, List,  Dict, Callable


class PrecisionError(Exception):
    """== PrecisionError *(Exception Class)*==

    This class defines an exception meant to be raised when any type of rounding or loss of precision that causes the time coordinates of a PWL object to not be strictly increasing.
    """


class PWL():

    _dict_of_objects: Dict = {}

    def __init__(self, t_step: float = 1e-9, name: Optional[str] = None, verbose: bool = False) -> None:

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
        self._t_list: List[float] = []
        self._x_list: List[float] = []
        self._t_step: float = t_step
        self._name: str = name
        self._verbose: bool = verbose

        if name in PWL._dict_of_objects:
            raise ValueError(f"Name '{name}' already in use.")

        PWL._dict_of_objects[name] = self

    def __str__(self) -> str:

        return f"{self.name}: PWL object with {len(self.t_list)} points and duration of {self.t_list[-1]} seconds"

    @property
    def t_list(self) -> List[float]:

        return self._t_list

    @property
    def x_list(self) -> List[float]:

        return self._x_list

    @property
    def t_step(self) -> float:

        return self._t_step

    @t_step.setter
    def t_step(self, new_t_step: float) -> None:

        if not isinstance(new_t_step, Real):
            raise TypeError(
                f"Attribute 't_step' should be a real number but has type '{type(new_t_step).__name__}'.")
        if new_t_step <= 0:
            raise ValueError(
                f"Attribute 't_step' should be strictly positive but has value of {new_t_step}.")

        self._t_step = new_t_step

    @property
    def name(self) -> str:

        return self._name

    @name.setter
    def name(self, new_name: str) -> None:

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

        return self._verbose

    @verbose.setter
    def verbose(self, new_verbose: bool) -> None:

        if not isinstance(new_verbose, bool):
            raise TypeError(
                f"Attribute 'verbose' should be a boolean but has type '{type(new_verbose).__name__}'.")
        self._verbose = new_verbose

    def hold(self, duration: float) -> None:

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

        for t in np.arange(last_t+t_step, last_t+duration, t_step):
            self._add(t, f(t))

        self._add(last_t+duration, target)

    def _add(self, t: float, x: float) -> None:

        if len(self._t_list) >= 1 and t <= self._t_list[-1]:
            raise PrecisionError(
                f"Internal Python rounding caused the time coordinates to not be strictly increasing when adding points to {self._name}.")

        if len(self._t_list) == len(self._x_list) < 2:
            self._t_list.append(t)
            self._x_list.append(x)
        else:
            self._add_with_redundancy_check(x, t)

    def _add_with_redundancy_check(self, x: float, t: float):

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

        for t in np.arange(last_t+t_step, last_t+duration, t_step):
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

        for t in np.arange(last_t+t_step, last_t+duration, t_step):
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
                        "The chosen precision level caused the written time coordinates to not be strictly increasing.")
                file.write(f"{ti_str}    {xi_str}\n")
                last_t = ti_str


def _exp_edge_func(tau: float, t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:
    """== _exp_edge_func *(Private Function)* ==

    Private function that generates a function of the following form:

        f(t) = A + B*exp(-t/tau)

    The constants `A` and `B` chosen such that the conditions following conditions are met:

        f(t1) = f1
        f(t2) = f2
    """

    A = (f1*np.exp(t1/tau) - f2*np.exp(t2/tau)) / \
        (np.exp(t1/tau) - np.exp(t2/tau))
    B = (f1 - f2)/(np.exp(-t1/tau) - np.exp(-t2/tau))

    def f(t: float) -> float:
        return A + B*np.exp(-t/tau)

    return f


def _sin_edge_func(t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:
    """== _sin_edge_func *(Private Function)* ==

    Private function that generates a function of the following form:

        f(t) = A*sin(w*t - phi)

    The constants `A`, `w` and `phi` are chosen such that the following conditions are met:

        f(t1) = f1
        f(t2) = f2
        f'(t1) = f'(t2) = 0

    Due to the periodic nature of sine, infinite solutions exist. The solution with smallest `w` is chosen.
    """

    fm = (f1+f2)/2
    tm = (t1+t2)/2
    T = 2*(t2-t1)
    w = 2*np.pi/T
    phi = w*tm
    A = f2-fm

    def f(t: float) -> float:
        return fm + A*np.sin(w*t - phi)

    return f


def _smoothstep_edge_func(t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:
    """== _smoothstep_edge_func *(Private Funtion)* ==

    Private function that generates a function of the following form:

        f(t) = A + B*t + C*t^2 + D*t^3

    The constants `A`, `B`, `C` and `D` are chosen such that the following conditions are met:

        f(t1) = f1
        f(t2) = f2
        f'(t1) = f'(t2) = 0
    """

    Am = np.array([[1, t1, t1**2, t1**3],
                   [1, t2, t2**2, t2**3],
                   [0, 1, 2*t1, 3*t1**2],
                   [0, 1, 2*t2, 3*t2**2]])
    Bm = np.array([f1, f2, 0, 0])
    A, B, C, D = np.linalg.solve(Am, Bm)

    def f(t):
        return A + B*t + C*t**2 + D*t**3

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
    print(pwl)
