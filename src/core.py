# [Custom steps](https://distilabel.argilla.io/dev/sections/how_to_guides/basic/step/)

import time
from typing import Any, Dict, List
import pandas as pd
import os
from distilabel.steps import Step, StepInput, StepOutput
from distilabel.steps.tasks import TextGeneration, UltraFeedback

class ThrottledTextGeneration(TextGeneration):
    """
    A custom `TextGeneration` step that enforces a delay before processing each input.

    This class inherits from `distilabel.steps.tasks.TextGeneration` and overrides
    the `format_input` method to introduce a synchronous `time.sleep` operation.
    This is particularly useful for managing API rate limits when interacting
    with LLM providers.

    Attributes:
        delay (float): The duration in seconds to pause execution before formatting
                       and sending the input to the underlying LLM. Defaults to 12.0 seconds.
    """
    # https://github.com/argilla-io/distilabel/blob/313fac85b1a2472dd88db1a31c2b754599f46476/src/distilabel/steps/tasks/text_generation.py#L29-L46
    delay: float = 12.0

    def format_input(self, input: Dict[str, Any]) -> Any:
        print(f"⏳ [ThrottledTextGeneration] Throttling: sleeping {self.delay}s...")
        time.sleep(self.delay)

        return super().format_input(input)


class ThrottledUltraFeedback(UltraFeedback):
    """
    A custom `UltraFeedback` step that enforces a delay before processing each input.

    This class inherits from `distilabel.steps.tasks.UltraFeedback` and overrides
    the `process` method to introduce a synchronous `time.sleep` operation.
    This is crucial for managing API rate limits, especially for judge models
    that may have strict Tokens Per Minute (TPM) or Requests Per Minute (RPM) limits.

    Attributes:
        delay (float): The duration in seconds to pause execution before processing
                       a batch of inputs. Defaults to 31.0 seconds.
    """
    delay: float = 31.0

    # We MUST use StepInput (which is an alias for the Annotated list) so distilabel can validate the pipeline connection.
    def process(self, inputs: StepInput) -> StepOutput:
        print(f"⏳ [ThrottledUltraFeedback] Throttling: sleeping {self.delay}s...")
        time.sleep(self.delay)

        return super().process(inputs)

class CheckpointStep(Step):
    """
    A step that saves intermediate results to a CSV file to prevent data loss.

    This custom `distilabel` step is designed to periodically save processed
    data to a CSV file. This acts as a checkpointing mechanism, allowing
    the pipeline to resume from the last saved state in case of interruptions
    or crashes. It appends new data to the file, preserving previously
    processed entries.

    Attributes:
        file_path (str): The path to the CSV file where the checkpoints will be saved.
                         Defaults to "generation_checkpoint.csv".
    """

    file_path: str = "generation_checkpoint.csv"

    @property
    def inputs(self) -> List[str]:
        """Defines the expected input columns for this step. Returns an empty list as it accepts any input."""
        return []

    @property
    def outputs(self) -> List[str]:
        """Defines the output columns produced by this step. Returns an empty list as it passes through inputs."""
        return []

    def process(self, *inputs: StepInput) -> StepInput:
        """
        Processes the input batches, saves them to a CSV, and yields them for the next step.

        Args:
            *inputs: Variable length argument list of `StepInput` (list of dictionaries)
                     representing batches of data to be processed.

        Yields:
            StepInput: The original input batch, passed through to the next step in the pipeline.
        """
        for input_batch in inputs:
            df = pd.DataFrame(input_batch)
            file_exists = os.path.isfile(self.file_path)
            # Append to CSV, adding header only if the file doesn't exist yet
            df.to_csv(self.file_path, mode='a', index=False, header=not file_exists)
            print(f"💾 [CheckpointStep] Saved {len(df)} rows to {self.file_path}")

            yield input_batch
