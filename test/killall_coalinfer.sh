kill -9 $(ps aux | grep 'coalinfer.launch_server' | grep -v 'grep' | awk '{print $2}')
