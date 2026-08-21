from __future__ import annotations

import os

from lightning.pytorch.loggers import (
    CSVLogger,
)


def build_logger(
    cfg,
    run_dir: str,
):
    # Remove deprecated / obsolete anonymous W&B mode.
    os.environ.pop(
        "WANDB_ANONYMOUS",
        None,
    )

    wcfg = getattr(
        cfg,
        "wandb",
        None,
    )

    enabled = (
        bool(
            getattr(
                wcfg,
                "enable",
                False,
            )
        )
        if wcfg is not None
        else False
    )

    if enabled:
        try:
            from lightning.pytorch.loggers import (
                WandbLogger,
            )

            mode = str(
                getattr(
                    wcfg,
                    "mode",
                    "online",
                )
            )

            os.environ[
                "WANDB_MODE"
            ] = mode

            return WandbLogger(
                project=str(
                    wcfg.project
                ),
                entity=getattr(
                    wcfg,
                    "entity",
                    None,
                ),
                group=getattr(
                    wcfg,
                    "group",
                    None,
                ),
                name=(
                    getattr(
                        wcfg,
                        "name",
                        None,
                    )
                    or str(
                        cfg.experiment.name
                    )
                ),
                tags=list(
                    getattr(
                        wcfg,
                        "tags",
                        [],
                    )
                ),
                save_dir=run_dir,
                log_model=bool(
                    getattr(
                        wcfg,
                        "log_model",
                        False,
                    )
                ),
            )

        except Exception as exc:
            print(
                "[wandb] unavailable; "
                "falling back to CSVLogger: "
                f"{exc}"
            )

    return CSVLogger(
        save_dir=run_dir,
        name="logs",
    )