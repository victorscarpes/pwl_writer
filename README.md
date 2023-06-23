# pwl_writer

*by Victor Sabiá P. Carpes*

Tested on python version `3.6.6` with numpy version `1.19.5` and python version `3.11.1` with numpy version `1.25.0`. Type stubs for older numpy versions for mypy checking can be found [here](https://github.com/numpy/numpy-stubs).

## Summary

This package defines a class `PWL` to generate objects that represent time dependent signals $x(t)$ that need to be coded in a PWL file. Those objects are built using little components such as rectangular pulses and sawtooth pulses that can be chained together.

## Motivation

The motivations for this package are the nuisances of writing PWL files by hand. To properly explain this, let's discuss how PWL files work.

PWL stands for piecewise linear. A PWL file is a way to represent a time dependent signal (referenced by $x(t)$ from now on) for simulation softwares such as LTspice and Cadence Virtuoso. In it, points of the form $(t, x)$ are stored. During simulation, those points are interpolated with first degree polynomials. This poses 2 problems:

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

```python
mode1_state(3)
idle_state(1)
mode2_state(5)
```

---

## How to Install

To install, simply run one of the following commands:

```
pip install git+https://github.com/victorscarpes/pwl_writer
pip3 install git+https://github.com/victorscarpes/pwl_writer
```

## Documentation

Full documentation for the package can be found [here](https://htmlpreview.github.io/?https://github.com/victorscarpes/pwl_writer/blob/main/docs/pwl_writer.html). It was written with [Pycco](https://github.com/pycco-docs/pycco).
