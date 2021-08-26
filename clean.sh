#!/usr/bin/env bash

crm () {
    if [[ -e $1 ]]
    then
        rm -r $1
    fi
}


crm __pycache__/
crm dist/
crm mpi_channels.egg-info/
