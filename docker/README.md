# Docker images

## Frontend

Read-only React + FastAPI viewer for `pipelines` run/registration
bookkeeping - see the root README's "Frontend" section for what it shows.

For local dev, prefer `make pipelines.up` (from the repository root) - it
brings up Postgres, MinIO, and this frontend together, wired to talk to
each other. The targets below are for building/publishing a standalone,
versioned image instead.

To build the frontend docker image, run

`make frontend.build`

To tag and push the frontend docker image, run

`make frontend.push`

To run the built image standalone (pointed at your own Postgres/S3), run

`make frontend.run`
