import click

import logging


@click.group()
@click.option("-v", "--verbose", default=0, count=True)
def moosenet(verbose):
    """CLI for moosenet"""
    verbose = min(verbose, 2)
    count2level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=count2level[verbose],
    )
