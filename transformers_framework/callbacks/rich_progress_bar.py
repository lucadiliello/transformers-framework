from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, cast

import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar as _RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import CustomProgress, MetricsTextColumn, RichProgressBarTheme
from rich import get_console, reconfigure
from rich.progress import Task, TaskID
from rich.style import Style
from rich.text import Text


@dataclass
class CustomRichProgressBarTheme(RichProgressBarTheme):
    metrics_keys: Union[str, Style] = "white"


theme = CustomRichProgressBarTheme(
    description="green_yellow",
    progress_bar="green",
    progress_bar_finished="green",
    progress_bar_pulse="#6206E0",
    batch_progress="green_yellow",
    time="#99CCFF",
    processing_speed="#E5CCFF",
    metrics="#E5FFCC",
    metrics_keys="#FF9999",
)


class CustomMetricsTextColumn(MetricsTextColumn):

    def __init__(
        self,
        trainer: "pl.Trainer",
        style: Union[str, "Style"],
        keys_style: Union[str, "Style"],
        text_delimiter: str,
        metrics_format: str,
    ):
        super().__init__(trainer, style, text_delimiter, metrics_format)
        self._keys_style = keys_style

    def render(self, task: "Task") -> Text:
        assert isinstance(self._trainer.progress_bar_callback, RichProgressBar)  # nosec
        if (
            self._trainer.state.fn != "fit"
            or self._trainer.sanity_checking
            or self._trainer.progress_bar_callback.train_progress_bar_id != task.id
        ):
            return Text()
        if self._trainer.training and task.id not in self._tasks:
            self._tasks[task.id] = "None"
            if self._renderable_cache:
                self._current_task_id = cast(TaskID, self._current_task_id)
                self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]
            self._current_task_id = task.id
        if self._trainer.training and task.id != self._current_task_id:
            return self._tasks[task.id]

        text = ""
        for k, v in self._metrics.items():
            text += f"{k}: {round(v, 3) if isinstance(v, float) else v} "
        res = Text(text, justify="left", style=self._style)
        res.highlight_words(self._metrics.keys(), style=self._keys_style)
        return res


class RichProgressBar(_RichProgressBar):

    def __init__(
        self,
        refresh_rate: int = 1,
        leave: bool = True,
        theme: RichProgressBarTheme = theme,
        console_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(refresh_rate=refresh_rate, leave=leave, theme=theme, console_kwargs=console_kwargs)

    def _update_metrics(self, trainer, pl_module) -> None:
        metrics = self.get_metrics(trainer, pl_module)
        metrics['global_step'] = trainer.global_step
        metrics.pop('v_num')
        if self._metric_component:
            self._metric_component.update(metrics)

    def _init_progress(self, trainer: "pl.Trainer") -> None:
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            self._console.clear_live()
            self._metric_component = CustomMetricsTextColumn(
                trainer,
                self.theme.metrics,
                self.theme.metrics_keys,
                self.theme.metrics_text_delimiter,
                self.theme.metrics_format,
            )
            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False
