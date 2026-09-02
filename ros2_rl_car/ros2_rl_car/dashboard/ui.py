"""Live Tk/Matplotlib training and greedy-policy dashboard."""

from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import ttk

import numpy as np

from .telemetry import LiveTelemetry, TelemetryStore


class TrainingDashboard:
    """Continuously refreshed, vertically scrollable training dashboard."""

    def __init__(
        self,
        telemetry: TelemetryStore,
        track_points: np.ndarray,
        *,
        on_pause: Callable[[bool], None],
        on_save: Callable[[], None],
        on_greedy: Callable[[bool], None],
        refresh_ms: int = 250,
    ) -> None:
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError as exc:
            raise RuntimeError("GUI mode needs Tkinter and Matplotlib") from exc

        self._store = telemetry
        self._track = np.asarray(track_points, dtype=float)
        self._on_pause = on_pause
        self._on_save = on_save
        self._on_greedy = on_greedy
        self._refresh_ms = refresh_ms
        self.root = tk.Tk()
        self.root.title("ROS 2 RL Car — Live PPO Training")
        self.root.geometry("1280x900")
        self.root.minsize(900, 650)

        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True)
        scroll = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=scroll.yview)
        scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        content = ttk.Frame(scroll)
        window = scroll.create_window((0, 0), window=content, anchor=tk.NW)
        content.bind(
            "<Configure>", lambda _e: scroll.configure(scrollregion=scroll.bbox("all"))
        )
        scroll.bind("<Configure>", lambda e: scroll.itemconfigure(window, width=e.width))
        scroll.bind_all("<MouseWheel>", lambda e: scroll.yview_scroll(-e.delta // 120, "units"))

        toolbar = ttk.Frame(content, padding=6)
        toolbar.pack(fill=tk.X)
        self._pause_text = tk.StringVar(value="Pause")
        self._greedy_text = tk.StringVar(value="Watch greedy: off")
        ttk.Button(toolbar, textvariable=self._pause_text, command=self._toggle_pause).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Save checkpoint", command=on_save).pack(side=tk.LEFT, padx=8)
        ttk.Button(toolbar, textvariable=self._greedy_text, command=self._toggle_greedy).pack(side=tk.LEFT)
        self._status = ttk.Label(toolbar, text="Waiting for sensor frames…")
        self._status.pack(side=tk.RIGHT)

        figure = Figure(figsize=(12, 9), dpi=100, constrained_layout=True)
        grid = figure.add_gridspec(3, 3)
        self._track_ax = figure.add_subplot(grid[:2, :2])
        self._episode_ax = figure.add_subplot(grid[0, 2])
        self._loss_ax = figure.add_subplot(grid[1, 2])
        self._action_ax = figure.add_subplot(grid[2, 0])
        self._value_ax = figure.add_subplot(grid[2, 1])
        self._length_ax = figure.add_subplot(grid[2, 2])
        canvas = FigureCanvasTkAgg(figure, master=content)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6)
        self._canvas = canvas

        lower = ttk.Panedwindow(content, orient=tk.HORIZONTAL)
        lower.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        hyper_frame = ttk.LabelFrame(lower, text="Hyperparameters")
        log_frame = ttk.LabelFrame(lower, text="Event log")
        lower.add(hyper_frame, weight=1)
        lower.add(log_frame, weight=2)
        self._hyper = tk.Text(hyper_frame, height=12, width=40, state=tk.DISABLED)
        self._hyper.pack(fill=tk.BOTH, expand=True)
        self._log = tk.Text(log_frame, height=12, state=tk.DISABLED, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, command=self._log.yview)
        self._log.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._log.pack(fill=tk.BOTH, expand=True)
        self._last_event_count = 0
        self.root.after(0, self._refresh)

    def _toggle_pause(self) -> None:
        paused = not self._store.snapshot().paused
        self._on_pause(paused)
        self._pause_text.set("Resume" if paused else "Pause")

    def _toggle_greedy(self) -> None:
        greedy = not self._store.snapshot().greedy
        self._on_greedy(greedy)
        self._greedy_text.set(f"Watch greedy: {'on' if greedy else 'off'}")

    @staticmethod
    def _running_mean(values: np.ndarray, width: int = 20) -> np.ndarray:
        if values.size == 0:
            return values
        result = np.empty_like(values, dtype=float)
        for index in range(values.size):
            result[index] = values[max(0, index - width + 1) : index + 1].mean()
        return result

    def _refresh(self) -> None:
        state = self._store.snapshot()
        self._draw_track(state)
        self._draw_episodes(state)
        self._draw_updates(state)
        self._draw_policy(state)
        self._draw_text(state)
        self._canvas.draw_idle()
        self._status.configure(
            text=(
                f"episodes {len(state.episodes)} | updates {len(state.updates)} | "
                f"{'finished' if state.finished else 'training'}"
            )
        )
        if not state.finished:
            self.root.after(self._refresh_ms, self._refresh)

    def _draw_track(self, state: LiveTelemetry) -> None:
        ax = self._track_ax
        ax.clear()
        ax.set_title("Live top-down trajectory and lidar")
        if self._track.size:
            closed = np.vstack((self._track, self._track[0]))
            ax.plot(closed[:, 0], closed[:, 1], "k--", lw=1, label="centre line")
        if state.trajectory.size:
            ax.plot(state.trajectory[:, 0], state.trajectory[:, 1], "C0", lw=1, label="trajectory")
        x, y, heading = state.car_pose
        ax.arrow(x, y, 0.4 * np.cos(heading), 0.4 * np.sin(heading), width=0.04, color="C3")
        if state.ray_endpoints.size:
            endpoints = state.ray_endpoints
            for endpoint in endpoints[:: max(1, len(endpoints) // 40)]:
                ax.plot([x, endpoint[0]], [y, endpoint[1]], color="C2", alpha=0.22, lw=0.6)
        ax.axis("equal")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    def _draw_episodes(self, state: LiveTelemetry) -> None:
        ax = self._episode_ax
        ax.clear()
        ax.set_title("Episode reward / success")
        rewards = np.asarray([item.reward for item in state.episodes])
        if rewards.size:
            x = np.arange(1, rewards.size + 1)
            ax.plot(x, rewards, alpha=0.35, label="reward")
            ax.plot(x, self._running_mean(rewards), label="mean(20)")
            successes = np.asarray([item.success for item in state.episodes], dtype=float)
            ax.plot(x, self._running_mean(successes) * max(1.0, abs(rewards).max()), label="success rate × scale")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

        length_ax = self._length_ax
        length_ax.clear()
        length_ax.set_title("Episode length")
        lengths = [item.length for item in state.episodes]
        if lengths:
            length_ax.plot(lengths, color="C4")
        length_ax.grid(alpha=0.2)

    def _draw_updates(self, state: LiveTelemetry) -> None:
        ax = self._loss_ax
        ax.clear()
        ax.set_title("PPO update metrics")
        if state.updates:
            for attribute, label in (
                ("policy_loss", "policy"), ("value_loss", "value"),
                ("entropy", "entropy"), ("grad_norm", "grad norm"),
                ("approx_kl", "KL"), ("clip_fraction", "clip frac"),
            ):
                ax.plot([getattr(item, attribute) for item in state.updates], label=label)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.2)

    def _draw_policy(self, state: LiveTelemetry) -> None:
        ax = self._action_ax
        ax.clear()
        ax.set_title("Current action distribution")
        probabilities = state.action_probabilities
        if probabilities:
            labels = ("left", "straight", "right")[: len(probabilities)]
            ax.bar(labels, probabilities)
            ax.set_ylim(0.0, 1.0)
        value_ax = self._value_ax
        value_ax.clear()
        value_ax.set_title("Current V(s)")
        value_ax.bar(["value"], [state.value_estimate], color="C5")
        value_ax.axhline(0.0, color="k", lw=0.5)

    def _draw_text(self, state: LiveTelemetry) -> None:
        self._hyper.configure(state=tk.NORMAL)
        self._hyper.delete("1.0", tk.END)
        self._hyper.insert(tk.END, "\n".join(f"{key}: {value}" for key, value in state.hyperparameters.items()))
        self._hyper.configure(state=tk.DISABLED)
        if len(state.events) != self._last_event_count:
            self._log.configure(state=tk.NORMAL)
            self._log.delete("1.0", tk.END)
            self._log.insert(tk.END, "\n".join(state.events))
            self._log.see(tk.END)
            self._log.configure(state=tk.DISABLED)
            self._last_event_count = len(state.events)

    def run(self) -> None:
        self.root.mainloop()


def launch_dashboard(*args: object, **kwargs: object) -> None:
    TrainingDashboard(*args, **kwargs).run()
