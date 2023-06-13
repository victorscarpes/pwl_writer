import numpy
from collections.abc import Callable


class PrecisionError(Exception):
    pass


def _exp_edge_func(tau: float, t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:

    A = (f1*numpy.exp(t1/tau) - f2*numpy.exp(t2/tau))/(numpy.exp(t1/tau) - numpy.exp(t2/tau))
    B = (f1 - f2)/(numpy.exp(-t1/tau) - numpy.exp(-t2/tau))

    def f(t: float) -> float:
        return A + B*numpy.exp(-t/tau)

    return f


def _sin_edge_func(t1: float, f1: float, t2: float, f2: float) -> Callable[[float], float]:

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

    Am = numpy.array([[1, t1, t1**2, t1**3], [1, t2, t2**2, t2**3], [0, 1, 2*t1, 3*t1**2], [0, 1, 2*t2, 3*t2**2]])
    Bm = numpy.array([f1, f2, 0, 0])
    A, B, C, D = numpy.linalg.solve(Am, Bm)

    def f(t):
        return A + B*t + C*t**2 + D*t**3

    return f


class PWL():

    _list_of_names: list[str] = []

    def __init__(self, t_step: float = 1e-9, name: str | None = None, verbose: bool = False) -> None:
        """
        Initializer for the PWL class.

        Args:
            t_step (float, optional): Default time step for all operations. Should be strictly positive. Defaults to 1e-9.
            name (str | None, optional): Name of the PWL object. If set to None, autogenerates name. Should not be empty. Defaults to None.
            verbose (bool, optional): Flag that controls if verbose output sgould be printed during operations. Defaults to False.

        Raises:
            TypeError: Raised if any of the arguments has an invalid type.
            ValueError: Raised if any of the arguments has an invalid value.
        """

        # Check for nullable arguments
        if name is None:
            i: int = 0
            while f"pwl{i}" in PWL._list_of_names:
                i += 1
            name = f"pwl{i}"

        # Check type of arguments
        if not isinstance(t_step, (int, float)):
            raise TypeError(f"Argument 't_step' should be a real number but has type '{type(t_step).__name__}'.")
        if not isinstance(name, str):
            raise TypeError(f"Argument 'name' should either be a string or be None but has type '{type(name).__name__}'.")
        if not isinstance(verbose, bool):
            raise TypeError(f"Argument 'verbose' should be a boolean but has type '{type(verbose).__name__}'.")

        # Check value of arguments
        if t_step <= 0:
            raise ValueError(f"Argument 't_step' should be strictly positive but has value of {t_step}.")
        if not name:
            raise ValueError("Argument 'name' should not be empty.")

        # Actual function
        self.t_list: list[float] = []
        self.x_list: list[float] = []
        self.t_step = t_step
        self.name = name
        self.verbose = verbose
        PWL._list_of_names.append(name)

    def _add(self, t: float, x: float) -> None:
        if len(self.t_list) >= 1 and t <= self.t_list[-1]:
            raise PrecisionError(f"Internal Python rounding caused the time coordinates to not be strictly increasing when adding values to {self.name}.")

        if len(self.t_list) == len(self.x_list) < 2:
            self.t_list.append(t)
            self.x_list.append(x)
        else:
            t_n_1 = self.t_list[-1]
            t_n_2 = self.t_list[-2]

            x_n_1 = self.x_list[-1]
            x_n_2 = self.x_list[-2]

            last_m = (x_n_1 - x_n_2)/(t_n_1 - t_n_2)
            new_m = (x - x_n_1)/(t - t_n_1)

            if last_m == new_m:
                self.t_list[-1] = t
                self.x_list[-1] = x
            else:
                self.t_list.append(t)
                self.x_list.append(x)

    def hold(self, duration: float) -> None:
        """
        Method that takes the last value of PWL object and holds it for a given duration.

        Args:
            duration (float): Duration to hold for. Should be strictly positive.

        Raises:
            TypeError: Raised if any of the arguments has an invalid type.
            ValueError: Raised if any of the arguments has an invalid value.
        """

        # Check type of arguments
        if not isinstance(duration, (int, float)):
            raise TypeError(f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")

        # Check value of arguments
        if duration <= 0:
            raise ValueError(f"Argument 'duration' should be strictly positive but has value of {duration}.")

        # Actual function
        if self.verbose:
            print(f"{self.name}: Adding hold with duration of {duration}.")

        if len(self.t_list) == len(self.x_list) == 0:
            if self.verbose:
                print(f"{self.name}: Empty PWL object. Adding initial (0, 0) point.")
            self._add(0, 0)

        last_t = self.t_list[-1]
        last_x = self.x_list[-1]

        self._add(last_t+duration, last_x)

    def square_pulse(self, value: float, duration: float, t_step: float | None = None) -> None:
        """
        Method that generates a square pulse with given value and duration.

        If the duration is less than or equal to the time step, a linear transition with duration equal to the timestep is generated instead.

        Args:
            value (float): Amplitude of the square pulse.
            duration (float): Duration of the pulse. Should be strictly positive.
            t_step (float | None, optional): Transition time at discontinuity. Should be strictly positive. If set to None, uses the value stored on the PWL object. Defaults to None.

        Raises:
            TypeError: Raised if any of the arguments has an invalid type.
            ValueError: Raised if any of the arguments has an invalid value.
        """

        # Check for nullable arguments
        if t_step is None:
            t_step = self.t_step

        # Check type of arguments
        if not isinstance(value, (int, float)):
            raise TypeError(f"Argument 'value' should be a real number but has type '{type(value).__name__}'.")
        if not isinstance(duration, (int, float)):
            raise TypeError(f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")
        if not isinstance(t_step, (int, float)):
            raise TypeError(f"Argument 't_step' should either be a real number or be None but has type '{type(t_step).__name__}'.")

        # Check value of arguments
        if duration <= 0:
            raise ValueError(f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if t_step <= 0:
            raise ValueError(f"Argument 't_step' should be strictly positive but has value of {t_step}.")

        # Actual function
        if self.verbose:
            print(f"{self.name}: Adding square pulse with value of {value}, duration of {duration} and time step of {t_step}.")

        if duration <= t_step:
            if self.verbose:
                print(f"{self.name}: Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear edge.")
            self.lin_edge(value, t_step)
            return

        if len(self.t_list) == len(self.x_list) == 0:
            self._add(0, value)
        else:
            last_t = self.t_list[-1]
            self._add(last_t+t_step, value)

        last_t = self.t_list[-1]
        self._add(last_t+duration, value)

    def lin_edge(self, target: float, duration: float) -> None:
        """
        Method to generate a linear transition with a given duration from previous value to a given target. If the PWL object is empty, add the point (0, 0) and transitions from that to the target.

        Args:
            target (float): Value to transition to.
            duration (float): Duration of the transition. Should be strictly positive.

        Raises:
            TypeError: Raised if any of the arguments has an invalid type.
            ValueError: Raised if any of the arguments has an invalid value.
        """

        # Check type of arguments
        if not isinstance(target, (int, float)):
            raise TypeError(f"Argument 'target' should be a real number but has type '{type(target).__name__}'.")
        if not isinstance(duration, (int, float)):
            raise TypeError(f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")

        # Check value of arguments
        if duration <= 0:
            raise ValueError(f"Argument 'duration' should be strictly positive but has value of {duration}.")

        # Actual function
        if self.verbose:
            print(f"{self.name}: Adding linear edge with target of {target} and duration of {duration}.")

        if len(self.t_list) == len(self.x_list) == 0:
            if self.verbose:
                print(f"{self.name}: Empty PWL object. Adding initial (0, 0) point.")
            self._add(0, 0)

        last_t = self.t_list[-1]
        self._add(last_t+duration, target)

    def exp_edge(self, target: float, duration: float, tau: float, t_step: float | None = None) -> None:
        """
        Method to generate an exponential transition with a given duration from previous value to a given target.

        If the PWL object is empty, add the point (0, 0) and transitions from that to the target.

        The transition can either ease into the target or ease out of the previous value by setting the time constant to positive or negative respectively.

        If the duration is less than or equal to the time step, a linear transition with duration equal to the timestep is generated instead.

        Args:
            target (float): Value to transition to.
            duration (float): Duration of the transition. Should be strictly positive.
            tau (float): Time constant of the exponential. Should be non zero.
            t_step (float | None, optional): Minimal time step. Should be strictly positive. If set to None, uses the value stored on the PWL object. Defaults to None.

        Raises:
            TypeError: Raised if any of the arguments has an invalid type.
            ValueError: Raised if any of the arguments has an invalid value.
        """

        # Check for nullable arguments
        if t_step is None:
            t_step = self.t_step

        # Check type of arguments
        if not isinstance(target, (int, float)):
            raise TypeError(f"Argument 'target' should be a real number but has type '{type(target).__name__}'.")
        if not isinstance(duration, (int, float)):
            raise TypeError(f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")
        if not isinstance(tau, (int, float)):
            raise TypeError(f"Argument 'tau' should be a real number but has type '{type(tau).__name__}'.")
        if not isinstance(t_step, (int, float)):
            raise TypeError(f"Argument 't_step' should either be a real number or be None but has type '{type(t_step).__name__}'.")

        # Check value of arguments
        if duration <= 0:
            raise ValueError(f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if tau == 0:
            raise ValueError("Argument 'tau' should be non zero.")
        if t_step <= 0:
            raise ValueError(f"Argument 't_step' should be strictly positive but has value of {t_step}.")

        # Actual function
        if self.verbose:
            print(f"{self.name}: Adding exponential edge with target of {target}, time constant of {tau}, duration of {duration} and time step of {t_step}.")

        if duration <= t_step:
            if self.verbose:
                print(f"{self.name}: Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear edge.")
            self.lin_edge(target, t_step)
            return

        if len(self.t_list) == len(self.x_list) == 0:
            if self.verbose:
                print(f"{self.name}: Empty PWL object. Adding initial (0, 0) point.")
            self._add(0, 0)

        last_t = self.t_list[-1]
        last_x = self.x_list[-1]

        f = _exp_edge_func(tau=tau, t1=last_t, t2=last_t+duration, f1=last_x, f2=target)

        for t in numpy.arange(last_t+t_step, last_t+duration, t_step):
            self._add(t, f(t))

        self._add(last_t+duration, target)

    def sin_edge(self, target: float, duration: float, t_step: float | None = None) -> None:
        """
        Method to generate a half sine transition with a given duration from previous value to a given target.

        If the PWL object is empty, add the point (0, 0) and transitions from that to the target.

        If the duration is less than or equal to the time step, a linear transition with duration equal to the timestep is generated instead.

        Args:
            target (float): Value to transition to.
            duration (float): Duration of the transition. Should be strictly positive.
            t_step (float | None, optional): Minimal time step. Should be strictly positive. If set to None, uses the value stored on the PWL object. Defaults to None.

        Raises:
            TypeError: Raised if any of the arguments has an invalid type.
            ValueError: Raised if any of the arguments has an invalid value.
        """

        # Check for nullable arguments
        if t_step is None:
            t_step = self.t_step

        # Check type of arguments
        if not isinstance(target, (int, float)):
            raise TypeError(f"Argument 'target' should be a real number but has type '{type(target).__name__}'.")
        if not isinstance(duration, (int, float)):
            raise TypeError(f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")
        if not isinstance(t_step, (int, float)):
            raise TypeError(f"Argument 't_step' should either be a real number or be None but has type '{type(t_step).__name__}'.")

        # Check value of arguments
        if duration <= 0:
            raise ValueError(f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if t_step <= 0:
            raise ValueError(f"Argument 't_step' should be strictly positive but has value of {t_step}.")

        # Actual function
        if self.verbose:
            print(f"{self.name}: Adding sinusoidal edge with target of {target}, duration of {duration} and time step of {t_step}.")

        if duration <= t_step:
            if self.verbose:
                print(f"{self.name}: Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear edge.")
            self.lin_edge(target, t_step)
            return

        if len(self.t_list) == len(self.x_list) == 0:
            if self.verbose:
                print(f"{self.name}: Empty PWL object. Adding initial (0, 0) point.")
            self._add(0, 0)

        last_t = self.t_list[-1]
        last_x = self.x_list[-1]

        f = _sin_edge_func(t1=last_t, t2=last_t+duration, f1=last_x, f2=target)

        for t in numpy.arange(last_t+t_step, last_t+duration, t_step):
            self._add(t, f(t))

        self._add(last_t+duration, target)

    def smoothstep_edge(self, target: float, duration: float, t_step: float | None = None) -> None:
        """
        Method to generate a smoothstep polynomial transition with a given duration from previous value to a given target.

        If the PWL object is empty, add the point (0, 0) and transitions from that to the target.

        If the duration is less than or equal to the time step, a linear transition with duration equal to the timestep is generated instead.

        Args:
            target (float): Value to transition to.
            duration (float): Duration of the transition. Should be strictly positive.
            t_step (float | None, optional): Minimal time step. Should be strictly positive. If set to None, uses the value stored on the PWL object. Defaults to None.

        Raises:
            TypeError: Raised if any of the arguments has an invalid type.
            ValueError: Raised if any of the arguments has an invalid value.
        """

        # Check for nullable arguments
        if t_step is None:
            t_step = self.t_step

        # Check type of arguments
        if not isinstance(target, (int, float)):
            raise TypeError(f"Argument 'target' should be a real number but has type '{type(target).__name__}'.")
        if not isinstance(duration, (int, float)):
            raise TypeError(f"Argument 'duration' should be a real number but has type '{type(duration).__name__}'.")
        if not isinstance(t_step, (int, float)):
            raise TypeError(f"Argument 't_step' should either be a real number or be None but has type '{type(t_step).__name__}'.")

        # Check value of arguments
        if duration <= 0:
            raise ValueError(f"Argument 'duration' should be strictly positive but has value of {duration}.")
        if t_step <= 0:
            raise ValueError(f"Argument 't_step' should be strictly positive but has value of {t_step}.")

        # Actual function
        if self.verbose:
            print(f"{self.name}: Adding smoothstep edge with target of {target}, duration of {duration} and time step of {t_step}.")

        if duration <= t_step:
            if self.verbose:
                print(f"{self.name}: Duration of {duration} is less than or equal to time step of {t_step}. Converting to linear edge.")
            self.lin_edge(target, t_step)
            return

        if len(self.t_list) == len(self.x_list) == 0:
            if self.verbose:
                print(f"{self.name}: Empty PWL object. Adding initial (0, 0) point.")
            self._add(0, 0)

        last_t = self.t_list[-1]
        last_x = self.x_list[-1]

        f = _smoothstep_edge_func(t1=last_t, t2=last_t+duration, f1=last_x, f2=target)

        for t in numpy.arange(last_t+t_step, last_t+duration, t_step):
            self._add(t, f(t))

        self._add(last_t+duration, target)

    def write(self, filename: str, precision: int = 10) -> None:
        """
        Method to write the PWL object to a txt file. If file already exists, this methos overwrites it.

        Args:
            filename (str): Name of the file to be written.
            precision (int, optional): Significant figures to be used when writing. Should be strictly positive. Defaults to 10.

        Raises:
            TypeError: Raised if any of the arguments has an invalid type.
            ValueError: Raised if any of the arguments has an invalid value.
            PrecisionError: Raised if the chosen precision makes the written time coordinates not be strictly increasing.
        """

        # Check type of arguments
        if not isinstance(filename, str):
            raise TypeError(f"Argument 'filename' should be a string but has type '{type(filename).__name__}'.")
        if not isinstance(precision, int):
            raise TypeError(f"Argument 'precision' should be an integer but has type '{type(precision).__name__}'.")

        # Check value of arguments
        if precision <= 0:
            raise ValueError(f"Argument 'precision' should be strictly positive but has value of {precision}.")

        t_list = self.t_list
        x_list = self.x_list

        with open(filename, "w") as file:
            ti_str = numpy.format_float_scientific(t_list[0], precision-1, unique=False, sign=True)
            xi_str = numpy.format_float_scientific(x_list[0], precision-1, unique=False, sign=True)
            file.write(f"{ti_str}    {xi_str}\n")
            last_t = ti_str
            for ti, xi in zip(t_list[1:], x_list[1:]):
                ti_str = numpy.format_float_scientific(ti, precision-1, unique=False, sign=True)
                xi_str = numpy.format_float_scientific(xi, precision-1, unique=False, sign=True)
                if ti_str == last_t:
                    raise PrecisionError("The chosen precision level caused the written time coordinates to not be strictly increasing.")
                file.write(f"{ti_str}    {xi_str}\n")
                last_t = ti_str
