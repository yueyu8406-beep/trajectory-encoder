"""Process-local outbound-network prohibition for an entirely local experiment.

This is loaded before torch by the offline launcher and inherited by workers.
It does not change machine/user network or Python settings.
"""
import sys


def deny_outbound(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.gethostbyname", "socket.gethostbyaddr"):
        raise PermissionError("Encoder offline execution prohibits outbound sockets and DNS")


sys.addaudithook(deny_outbound)
