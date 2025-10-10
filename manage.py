#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
from dotenv import load_dotenv

load_dotenv("/home/naeem-md93/Shared/.env")
load_dotenv("/run/user/1000/gvfs/smb-share:server=an515-55.local,share=shared/.env")

import sys
from django.core.management.commands.runserver import Command as runserver


runserver.default_addr = os.getenv("AI_ADDR", "127.0.0.1")
runserver.default_port = os.getenv("AI_PORT", "8000")


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
