#!/bin/bash

OPENMPI_MPIRUN="/usr/bin/mpirun.openmpi"
OPENMPI_MPICXX="/usr/bin/mpicxx.openmpi"
INTEL_MPIRUN="/opt/intel/oneapi/mpi/latest/bin/mpirun"
INTEL_MPICXX="/opt/intel/oneapi/mpi/latest/bin/mpicxx"

show_help() {
    echo "Usage: $0 [openmpi|intel|check|help]"
    echo
    echo "Commands:"
    echo "  openmpi   Switch to OpenMPI"
    echo "  intel     Switch to Intel MPI"
    echo "  check     Show current MPI configuration"
    echo "  help      Show this help message"
    echo
    echo "Example:"
    echo "  $0 openmpi   # Switch to OpenMPI"
    echo "  $0 intel     # Switch to Intel MPI"
    echo "  $0 check     # Check current active MPI"
}

if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run as root (use sudo)"
    exit 1
fi

case "$1" in
    openmpi)
        if [ ! -f "$OPENMPI_MPIRUN" ] || [ ! -f "$OPENMPI_MPICXX" ]; then
            echo "❌ OpenMPI binaries not found at $OPENMPI_MPIRUN or $OPENMPI_MPICXX"
            exit 1
        fi
        update-alternatives --install /usr/bin/mpirun mpirun "$OPENMPI_MPIRUN" 50
        update-alternatives --install /usr/bin/mpicxx mpicxx "$OPENMPI_MPICXX" 50
        update-alternatives --set mpirun "$OPENMPI_MPIRUN"
        update-alternatives --set mpicxx "$OPENMPI_MPICXX"
        echo "✅ Switched to OpenMPI"
        ;;
    intel)
        if [ ! -f "$INTEL_MPIRUN" ] || [ ! -f "$INTEL_MPICXX" ]; then
            echo "❌ Intel MPI binaries not found at $INTEL_MPIRUN or $INTEL_MPICXX"
            exit 1
        fi
        update-alternatives --install /usr/bin/mpirun mpirun "$INTEL_MPIRUN" 60
        update-alternatives --install /usr/bin/mpicxx mpicxx "$INTEL_MPICXX" 60
        update-alternatives --set mpirun "$INTEL_MPIRUN"
        update-alternatives --set mpicxx "$INTEL_MPICXX"
        echo "✅ Switched to Intel MPI"
        ;;
    check)
        echo "Current MPI configuration:"
        echo "mpirun -> $(which mpirun)"
        mpirun --version | head -n 1
        echo "mpicxx -> $(which mpicxx)"
        mpicxx --version | head -n 1
        ;;
    help|"")
        show_help
        ;;
    *)
        echo "❌ Invalid command: $1"
        show_help
        exit 1
        ;;
esac
