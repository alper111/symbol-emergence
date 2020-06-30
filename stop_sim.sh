#!/usr/bin/zsh

echo "STOPPING ALL SERVERS"
kill $(cat out/server.pid)
rm out/server.pid out/torobo.out out/relay.out
