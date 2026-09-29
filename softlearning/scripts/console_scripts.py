"""A command line interface that exposes softlearning examples to user.

This module is exposed via the `mbpo` command (see setup.py entry_points).
Only local execution is supported; the ray-autoscaler cluster commands from
upstream softlearning were removed along with their `examples.instrument`
helpers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import click


from examples.instrument import (
    run_example_dry,
    run_example_local,
    run_example_debug)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.group()
def cli():
    pass


@cli.command(
    name='run_example_dry',
    context_settings={'ignore_unknown_options': True})
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
def run_example_dry_cmd(example_module_name, example_argv):
    """Print the variant spec and related information of an example."""
    return run_example_dry(example_module_name, example_argv)


@cli.command(
    name='run_local',
    context_settings={'ignore_unknown_options': True})
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
def run_example_local_cmd(example_module_name, example_argv):
    """Run example locally, potentially parallelizing across cpus/gpus."""
    return run_example_local(example_module_name, example_argv)


@cli.command(
    name='run_example_debug',
    context_settings={'ignore_unknown_options': True})
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
def run_example_debug_cmd(example_module_name, example_argv):
    """The debug mode limits tune trial runs to enable use of debugger."""
    return run_example_debug(example_module_name, example_argv)


cli.add_command(run_example_local_cmd)
cli.add_command(run_example_dry_cmd)

# Alias for run_example_local
cli.add_command(run_example_local_cmd, name='launch_example_local')
# Alias for run_example_dry
cli.add_command(run_example_dry_cmd, name='launch_example_dry')
# Alias for run_example_debug
cli.add_command(run_example_debug_cmd, name='launch_example_debug')


def main():
    return cli()


if __name__ == "__main__":
    main()
