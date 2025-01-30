from enum import Enum
import math
import textwrap
import warnings
from typing import Union, List, Tuple

from composer.core import State, Time, TimeUnit
from composer.optim.scheduler import (
    ComposerScheduler,
    LinearScheduler,
    _convert_time,
    _raise_if_max_duration_exceeds_t_max,
)


def _raise_if_schedule_and_max_incompatible(t_schedule: Time[int], t_max: Time[int], schedule_name: str):
    """Checks that t_schedule and t_max have the same units.

    _convert_time should be called on both `t_warmup` and `t_max` before this function is called. As a a result, t_warmup and t_max will not
    be TimeUnit.EPOCH.
    """
    assert (
        t_schedule.unit != TimeUnit.EPOCH and t_max.unit != TimeUnit.EPOCH
    ), "t_warmup and t_max cannot be in units of EPOCH"
    if isinstance(t_schedule, str):
        t_schedule = Time.from_timestring(t_schedule)
    if isinstance(t_max, str):
        t_max = Time.from_timestring(t_max)
    units_same = t_schedule.unit == t_max.unit
    if not units_same:
        raise ValueError(
            f"Cannot use {schedule_name} scheduler with units {t_schedule.unit} along with t_max "
            f"{t_max} with units {t_max.unit}. {schedule_name} and t_max must use the same units.",
        )


class Schedule(str, Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    INVERSE_SQRT = "inverse_sqrt"


def _linear_schedule(x: float, start_y: float = 1.0, finish_y: float = 0.0) -> float:
    """Implements a linear curve.

    Curve is linear(x) on domain [0, 1] and range [start_y, finish_y]. Additionally, param x is clipped to the interval [0, 1]
    """
    x = min(max(x, 0.0), 1.0)
    return start_y + (finish_y - start_y) * x


def _cosine_schedule(x: float, start_y: float = 1.0, finish_y: float = 0.0) -> float:
    """Implements a cosine curve.

    Curve is cos(x) on domain [0, pi], stretched to the domain [0, 1] and range [start_y, finish_y]. Additionally, param x is
    clipped to the interval [0, 1]
    """
    x = min(max(x, 0.0), 1.0)
    return finish_y + (start_y - finish_y) * (1 + math.cos(x * math.pi)) / 2


def _inverse_sqrt_schedule(x: Union[int, float], alpha: float = 1.0, beta: float = 0.0) -> float:
    """Implements an inverse square root curve.

    Curve is alpha / sqrt(x + beta). Additionally, param x is clipped to the interval [0, inf)
    """
    return alpha / math.sqrt(max(x, 0.0) + beta)


def _get_scheduler(scheduler_type: Schedule):
    if scheduler_type == Schedule.LINEAR:
        return _linear_schedule
    elif scheduler_type == Schedule.COSINE:
        return _cosine_schedule
    elif scheduler_type == Schedule.INVERSE_SQRT:
        return _inverse_sqrt_schedule
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")


class WarmupStableDecayScheduler(ComposerScheduler):
    r"""
    Args:
        t_warmup (str | Time): Warmup time.
        t_max (str | Time): The duration of this scheduler. Default = ``"1dur"``.
        alpha_f (float): Learning rate multiplier to decay to. Default = ``0.0``.
        scale_warmup (float): SSR also scales the warmup period. Default = ``False``.
    """

    def __init__(
        self,
        t_warmup: Union[str, Time],
        t_decay: Union[str, Time] = "0.1dur",
        t_max: Union[str, Time] = "1dur",
        alpha_f: float = 0.1,
        scale_warmup: bool = False,
    ):
        self.t_warmup = t_warmup
        self.t_decay = t_decay
        self.t_max = t_max
        self.alpha_f = alpha_f
        self.scale_warmup = scale_warmup
        self.warmup_scheduler = LinearScheduler(alpha_i=1e-10, alpha_f=1.0, t_max=t_warmup)

    def __call__(self, state: State, ssr: float = 1.0):
        assert state.max_duration is not None, "max_duration should be set whenever schedulers are invoked"
        t_warmup = _convert_time(self.t_warmup, state)
        t_decay = _convert_time(self.t_decay, state)
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent(
                    """\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""
                )
            )

        if state.timestamp < t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)
        elif state.timestamp < t_max - t_decay:
            return 1.0
        else:
            current_time = state.timestamp.get(t_warmup.unit)
            current_decay_time = current_time - (t_max - t_decay)
            current_decay_fraction = current_decay_time / t_decay
            lr_scale = 1 - current_decay_fraction.value * (1 - self.alpha_f)
            return max(0.0, lr_scale)  # prevent negative lr_scale


class CosineInverseSqrtScheduler(ComposerScheduler):
    r"""Decays the learning rate according to a cosine decay followed by an inverse square root decay, with optional warmup and cooldown.

    Specifically, the learning rate multiplier :math:`\gamma` (ignoring warmup and cooldown) can be expressed as:

    .. math::
        \alpha(t) = \begin{cases}
            \alpha_f + \frac{(1 - \alpha_f)}{2} \left[ \cos \left( \frac{1}{2}\pi \frac{t_{step}}{t_{cosine}} \right) + 1 \right], & \text{if } t_{step} \leq t_{cosine} \\
            \frac{\gamma}{\sqrt{t_{step} + \delta}}, & \text{if } t_{step} > t_{cosine}
        \end{cases}

    Where :math:`t_{cosine}` represents the half cosine decay duration, :math:`\alpha_f` represents the final learning rate multiplier, and
    :math:`\gamma` and :math:`\delta` are chosen to ensure continuity of the scheduler and its derivative.

    This scheduler is a more flexible version of the multi-stage infinite scheduler proposed in *Stable LM 2 1.6B Technical Report*
    <https://arxiv.org/abs/2402.17834>. To match that scheduler, keep the default of ``t_cosine="0.25dur"``.

    .. warning::
            By default, initial warmup, cosine, and final cooldown schedules are **not** scaled according to any provided scale schedule ratio.
            To change this behavior, set ``scale_schedules=True``.

    Args:
        t_warmup (str | Time): Warmup time.
        t_cooldown (str | Time): Cooldown time.
        t_cosine (str | Time): Half cosine decay duration. Default = ``"0.25dur"``.
        t_max (str | Time): The total duration of this scheduler. Default = ``"1dur"``.
        alpha_f (float): Final learning rate multiplier to decay to. Default = ``0.0``.
        alpha_s (float): Starting learning rate multiplier to warmup from. Default = ``0.0``.
        warmup_schedule (str | Schedule): Warmup schedule. Default = ``"linear"``.
        cooldown_schedule (str | Schedule): Cooldown schedule. Default = ``"linear"``.
        scale_schedules (float): SSR also scales the warmup, cosine, and cooldown periods. Default = ``False``.
    """

    def __init__(
        self,
        t_warmup: Union[str, Time],
        t_cooldown: Union[str, Time],
        t_cosine: Union[str, Time] = "0.25dur",
        t_max: Union[str, Time] = "1dur",
        alpha_f: float = 0.0,
        alpha_s: float = 0.0,
        warmup_schedule: Union[str, Schedule] = Schedule.LINEAR,
        cooldown_schedule: Union[str, Schedule] = Schedule.LINEAR,
        scale_schedules: bool = False,
    ):
        self.t_warmup = t_warmup
        self.t_cooldown = t_cooldown
        self.t_cosine = t_cosine
        self.t_max = t_max
        self.alpha_f = alpha_f
        self.alpha_s = alpha_s
        self.scale_schedules = scale_schedules
        self.gamma = None
        self.delta = None
        self._warmup_schedule = _get_scheduler(Schedule(warmup_schedule))
        self._cooldown_schedule = _get_scheduler(Schedule(cooldown_schedule))
        self._last_t_max = None

    def __call__(self, state: State, ssr: float = 1.0):
        assert state.max_duration is not None, "max_duration should be set whenever schedulers are invoked"
        t_warmup = _convert_time(self.t_warmup, state, ssr=ssr if self.scale_schedules else 1)
        t_cooldown = _convert_time(self.t_cooldown, state, ssr=ssr if self.scale_schedules else 1)
        t_cosine = _convert_time(self.t_cosine, state, ssr=ssr if self.scale_schedules else 1)
        t_max = _convert_time(self.t_max, state, ssr=ssr)

        if self._last_t_max != t_max:
            if t_warmup.value == 0:
                self._warmup_cooldown_warning("warmup")
            if t_cooldown.value == 0:
                self._warmup_cooldown_warning("cooldown")
            _raise_if_schedule_and_max_incompatible(t_warmup, t_max, "warmup")
            _raise_if_schedule_and_max_incompatible(t_cosine, t_max, "cosine")
            _raise_if_schedule_and_max_incompatible(t_cooldown, t_max, "cooldown")
            _raise_if_max_duration_exceeds_t_max(t_max, state)
            self._finish_setup(t_max, t_warmup, t_cosine, t_cooldown)
        else:
            self._last_t_max = t_max

        v_warmup = t_warmup.value
        v_cooldown = t_cooldown.value
        v_cosine = t_cosine.value
        v_max = t_max.value
        v_step = state.timestamp.get(t_max.unit).value

        if v_step < v_warmup:
            return self._warmup_schedule(v_step / v_warmup, start_y=self.alpha_s, finish_y=1)

        if v_step >= v_max - v_cooldown:
            return self._cooldown_schedule(
                (v_step - (v_max - v_cooldown)) / v_cooldown,
                start_y=self.cooldown_start,
                finish_y=self.alpha_f,
            )

        v_adjusted_step = v_step - v_warmup
        if v_adjusted_step <= v_cosine:
            return _cosine_schedule(0.5 * v_adjusted_step / v_cosine, finish_y=self.alpha_f)
        else:
            return _inverse_sqrt_schedule(v_adjusted_step, self.gamma, self.delta)

    def _finish_setup(self, t_max: Time[int], t_warmup: Time[int], t_cosine: Time[int], t_cooldown: Time[int]):
        self.delta = t_cosine.value * (1 + self.alpha_f - math.pi * (1 - self.alpha_f)) / (math.pi * (1 - self.alpha_f))
        self.gamma = (1 + self.alpha_f) / 2 * math.sqrt(t_cosine.value + self.delta)

        self.cooldown_start = _inverse_sqrt_schedule((t_max - t_cooldown - t_warmup).value, self.gamma, self.delta)

        if t_cooldown.value > 0 and self.alpha_f > self.cooldown_start:
            raise ValueError(
                f"The final learning rate multiplier alpha_f ({self.alpha_f}) must be less than the inverse "
                f"square root value at t_max - t_cooldown ({self.cooldown_start:.4f}) to ensure continuity."
            )

        self._last_t_max = t_max

    def _warmup_cooldown_warning(self, type: str):
        warnings.warn(
            textwrap.dedent(
                f"The {type} duration is 0. If you specified {type} as a fraction of total"
                f"training duration, take note that the {type} duration is calculated in the"
                f"same unit as the trainer's max_duration parameter."
            ),
        )


class OneMinusSqrtScheduler(ComposerScheduler):
    """
    Decays the learning rate according to a 1 - sqrt function, without an initial warmup phase.
    The learning rate decays from 1 to alpha_f according to the formula:
    f(n, N_T, N_C) = alpha_f + (1 - alpha_f)(1 - sqrt((n - (N_T - N_C)) / N_C))
    where N_T is t_max and N_C is t_decay.
    """

    def __init__(
        self,
        t_decay: Union[str, Time] = "0.1dur",
        t_max: Union[str, Time] = "1dur",
        alpha_f: float = 0.1,
    ):
        self.t_decay = t_decay
        self.t_max = t_max
        self.alpha_f = alpha_f

    def __call__(self, state: State, ssr: float = 1.0):
        assert state.max_duration is not None, "max_duration should be set whenever schedulers are invoked"

        t_decay = _convert_time(self.t_decay, state)
        t_max = _convert_time(self.t_max, state, ssr=ssr)

        current_time = state.timestamp.get(t_max.unit)

        if current_time <= t_max - t_decay:
            return 1.0
        else:
            relative_time = (current_time - (t_max - t_decay)) / t_decay
            lr_scale = self.alpha_f + (1 - self.alpha_f) * (1 - math.sqrt(relative_time.value))
            return max(self.alpha_f, lr_scale)


class CompositeScheduler(ComposerScheduler):
    """Combines multiple schedulers in sequence, each running for a specified duration.
    
    Args:
        schedulers: List of (scheduler, duration) tuples, where duration is a Time or str
            specifying how long that scheduler should run for. The last scheduler's duration 
            should be "remainder" to use remaining time.
        t_max (str | Time): Total duration of the schedule. Default = "1dur".
        scale_schedules (bool): Whether to scale scheduler durations with SSR. Default = False.
        
    Example:
        >>> scheduler = CompositeScheduler(
        ...     schedulers=[
        ...         (WarmupStableDecayScheduler(t_warmup="1000ba", alpha_f=1.0), "1000ba"),
        ...         (OneMinusSqrtScheduler(alpha_f=0.1), "remainder")
        ...     ],
        ...     t_max="10000ba"
        ... )
    """
    
    def __init__(
        self,
        schedulers: List[Tuple[ComposerScheduler, Union[str, Time]]],
        t_max: Union[str, Time] = "1dur",
        scale_schedules: bool = False,
    ):
        if not schedulers:
            raise ValueError("Must provide at least one scheduler")
            
        scheduler, duration = schedulers[-1]
        if duration != "remainder":
            raise ValueError("Last scheduler duration must be 'remainder'")
        
        # init the scheduler
        self.scheduler_specs = schedulers
        self.t_max = t_max
        self.scale_schedules = scale_schedules
        
    def _get_active_scheduler(
        self, 
        state: State,
        timestamp: Time,
        ssr: float = 1.0
    ) -> tuple[ComposerScheduler, float]:
        """Find the active scheduler and its start time."""
        current_time = 0
        
        # Check each scheduler except the last one
        for scheduler, duration in self.scheduler_specs[:-1]:
            if isinstance(duration, str):
                duration = _convert_time(duration, state, ssr=ssr if self.scale_schedules else 1.0)
            
            next_time = current_time + duration.value
            if timestamp.value < next_time:
                # Current scheduler is active
                return scheduler, current_time
                
            current_time = next_time
            
        # If we get here, use the last scheduler
        last_scheduler, _ = self.scheduler_specs[-1]
        return last_scheduler, current_time
        
    def __call__(self, state: State, ssr: float = 1.0) -> float:
        """Returns the learning rate multiplier for the current state."""
        assert state.max_duration is not None, "max_duration should be set whenever schedulers are invoked"
        
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_timestamp = state.timestamp.get(t_max.unit)
        assert t_max.unit == "tok", f"t_max must be in tokens, got {t_max.unit}"
        
        # Find active scheduler
        scheduler, start_time = self._get_active_scheduler(state, current_timestamp, ssr)
        
        # Convert current timestamp to scheduler's relative time
        relative_timestamp = Time(current_timestamp.value - start_time, current_timestamp.unit)
        
        new_timestamp_for_sub_scheduler = state.timestamp.copy(token=relative_timestamp.value)
        original_timestamp = state.timestamp
        state.timestamp = new_timestamp_for_sub_scheduler
        return_val = scheduler(state, ssr)
        state.timestamp = original_timestamp
        return return_val
        