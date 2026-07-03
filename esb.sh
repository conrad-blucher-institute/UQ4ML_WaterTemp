#!/bin/sh
# Bare `esb` launcher (bash / Grendal). `chmod +x esb.sh` then run ./esb.sh,
# or symlink it onto PATH:  ln -s "$(pwd)/esb.sh" ~/bin/esb
exec python -m esb "$@"
