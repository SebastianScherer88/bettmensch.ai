# Docker images

## Dashboard

To build the dashboard docker image, run

`make dashboard.build`

To tag and push the dashboard docker image, run

`make dashboard.push`

To run the dashboard container, run

`make dashboard.run`

## Pipeline SDK components

We have 3 off-the-shelve docker images to support the sdk's pipeline module:
- `base`
- `torch` (w/ GPU support)
- `lightning` (w/ GPU support)

To build a component docker image, run

`make component.build COMPONENT=<base|torch|lightning>`

To tag and push a component docker image, run

`make component.push COMPONENT=<base|torch|lightning>`