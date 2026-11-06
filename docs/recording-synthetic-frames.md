# Record synthetic frames

This example generates 100 monochrome images and records them using a
Dirigo pipeline. It requires no acquisition hardware.

The complete program is
[`examples/record_synthetic_frames.py`](../examples/record_synthetic_frames.py).

## Run the example

Activate your development environment and run the script from the
repository root:

```powershell
python examples/record_synthetic_frames.py
```

It creates `synthetic_frames.tif` in the current working directory and
prints the destination after recording completes.

The output contains one-hundred 256 × 256 unsigned 16-bit grayscale frames. 
Each frame contains a gradient, with brightness increasing between frames.

The writer preserves existing destinations. To repeat the example, remove
the previous output or change `destination` in the script.

## Define a source

`SyntheticFrameSource` subclasses `Source[FrameArray]`. It originates
image products and has no upstream worker.

A worker implements three lifecycle hooks:

| Hook | Purpose | Execution context |
|---|---|---|
| `_prepare()` | Prepare resources before execution | The thread calling `Pipeline.start()` |
| `_work()` | Produce or consume products | The worker's dedicated thread |
| `_close()` | Release resources, including after failures | The worker thread or pipeline caller during startup cleanup |

This source needs no external resources, so its preparation and cleanup
hooks contain only explanatory docstrings.

Its `_work()` method creates a base image pattern, then publishes 100
frames. Returning from `_work()` initiates normal worker cleanup and
closure of its output stream.

The underscore-prefixed hooks are the extension points used when
implementing a worker. Application code starts and waits for execution
through `Pipeline`.

## Publish pixels and metadata together

Each publication is a `Product` containing:

- `payload`: the image array.
- `metadata`: a `FrameMetadata` record identifying the frame.

The example sets `sequence_index` and `time_index` to the frame number,
starting at zero.

`sequence_index` identifies a publication in the sequence. `time_index`
identifies a time point. Neither field is an elapsed time or a request to
wait before publishing.

The source generates frames as quickly as downstream processing allows.

Metadata records are immutable. The image arrays remain mutable NumPy
arrays, so the source must leave them unchanged after publication. In this
example, each image addition creates a fresh array, and the source never
modifies that array again.

## Connect the writer

`TiffWriter` receives products from the source selected by its `upstream`
argument.

The example selects monochrome interpretation, which expects image arrays
with shape `(Y, X)`. The frame shape and dtype must remain constant
throughout a recording.

With the writer's default `contiguous=False`, each frame is stored as a
separate TIFF series with its own metadata. For these monochrome frames,
each series contains one page.

The writer creates the output when it receives its first valid frame. An
empty source would finish without creating an output file.

## Execute the pipeline

`Pipeline(writer)` includes the writer and its upstream source
automatically. Adding a terminal worker includes the chain of workers it
depends on.

`pipeline.start()` prepares the workers and launches execution. It returns
without waiting for the recording to finish.

`pipeline.wait()` waits for execution and cleanup to complete. It also
raises a worker failure in the calling thread, allowing the application
to report the error.

Both the workers and the pipeline are single-use. Another recording needs
new instances.

## Finish or cancel

On normal completion:

1. The source publishes its 100 products and returns from `_work()`.
2. Source cleanup closes its output stream.
3. The writer consumes the remaining queued products and reaches the end
   of the stream.
4. The writer closes its output resources.
5. `pipeline.wait()` returns.

The application therefore needs no separate call to close the TIFF writer.

The example also handles Ctrl+C by calling `pipeline.abort()` and then
`pipeline.wait()`. Aborting requests cancellation; waiting allows worker
cleanup to finish. A partially written TIFF file may remain.

## Buffer reuse

This example allocates a new array for each frame.

For sustained acquisition, a source can instead acquire reusable storage
from its output pool and publish a lease. Consumers must then respect the
borrowed payload's lifetime: the storage can be reused after its ownership
claims are released.

Buffer pooling is a separate topic that builds on the lifecycle and
publication behavior shown here.